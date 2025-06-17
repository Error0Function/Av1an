#[cfg(test)]
mod tests;

use std::{
    collections::HashMap,
    process::{exit, Command},
    str::FromStr,
};

use anyhow::{anyhow, bail, Result};
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while1},
    character::complete::{char, digit1, space0, space1},
    combinator::{map, map_res, opt, recognize, rest},
    error::Error as NomError,
    multi::separated_list0,
    sequence::{preceded, terminated},
    IResult, Parser,
};
use serde::{Deserialize, Serialize};

use crate::{
    context::Av1anContext,
    parse::valid_params,
    settings::{invalid_params, suggest_fix},
    Encoder,
};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Scene {
    pub start_frame: usize,
    pub end_frame: usize,
    pub zone_overrides: Option<ZoneOptions>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ZoneOptions {
    pub encoder: Encoder,
    pub passes: u8,
    pub video_params: Vec<String>,
    pub photon_noise: Option<u8>,
    pub extra_splits_len: Option<usize>,
    pub min_scene_len: usize,
}

type ParseResult<'a, O> = IResult<&'a str, O, NomError<&'a str>>;

// ======================== 辅助解析器 ========================

fn parse_number(i: &str) -> ParseResult<isize> {
    map_res(recognize((opt(char('-')), digit1)), |s: &str| s.parse()).parse(i)
}

fn parse_encoder(i: &str) -> ParseResult<Encoder> {
    map_res(
        alt((
            tag("aom"),
            tag("rav1e"),
            tag("x264"),
            tag("x265"),
            tag("vpx"),
            tag("svt-av1"),
        )),
        Encoder::from_str,
    )
    .parse(i)
}

fn parse_cli_arg(i: &str) -> ParseResult<(&str, Option<&str>)> {
    let key_parser = recognize((
        alt((tag("--"), tag("-"))),
        take_while1(|c: char| c.is_alphanumeric() || c == '-'),
    ));

    let value_parser = opt(preceded(
        alt((tag("="), space1)),
        take_while1(|c: char| !c.is_whitespace()),
    ));

    (key_parser, value_parser).parse(i)
}

// ======================== 重写后的主函数 ========================

impl Scene {
    pub fn parse_from_zone(input: &str, context: &Av1anContext) -> Result<Self> {
        // --- 阶段1: 解析 zone 的基本信息 ---
        let mut base_parser = (
            terminated(parse_number, space1),
            terminated(parse_number, space1),
            terminated(parse_encoder, space0),
            map(opt(terminated(tag("reset"), space0)), |r| r.is_some()),
            rest,
        );

        let (_, (start_frame_raw, end_frame_raw, encoder, reset, zone_args_str)) =
            base_parser.parse(input).map_err(|e: nom::Err<_>| {
                anyhow!("Failed to parse zone definition: {}", e.to_string())
            })?;

        let zone_args = zone_args_str.trim();

        // --- 验证和转换基本信息 (逻辑不变) ---
        let total_frames = context.frames;
        // 将 isize 转换为 usize，并处理 -1 的情况
        let start_frame = if start_frame_raw < 0 {
            bail!("Start frame cannot be negative.");
        } else {
            start_frame_raw as usize
        };
        let end_frame = if end_frame_raw == -1 {
            total_frames
        } else if end_frame_raw < 0 {
            bail!("End frame cannot be negative (except for -1).");
        } else {
            end_frame_raw as usize
        };

        if start_frame >= end_frame {
            bail!("Start frame must be earlier than the end frame");
        }
        if start_frame >= total_frames || end_frame > total_frames {
            bail!("Start and end frames must not be past the end of the video");
        }
        if encoder.format() != context.args.encoder.format() {
            bail!(
                "Zone specifies using {}, but this cannot be used in the same file as {}",
                encoder,
                context.args.encoder,
            );
        }
        if encoder != context.args.encoder {
            if encoder.get_format_bit_depth(context.args.output_pix_format.format).is_err() {
                bail!(
                    "Output pixel format {:?} is not supported by {} (used in zones file)",
                    context.args.output_pix_format.format,
                    encoder
                );
            }
            if !reset {
                bail!(
                    "Zone includes encoder change but previous args were kept. You probably meant \
                     to specify \"reset\"."
                );
            }
        }

        // --- 阶段2: 解析 zone 的参数覆盖 (逻辑不变) ---
        let mut video_params = if reset {
            Vec::new()
        } else {
            context.args.video_params.clone()
        };
        let mut passes = if reset {
            encoder.get_default_pass()
        } else {
            context.args.passes
        };
        let mut photon_noise = if reset {
            None
        } else {
            context.args.photon_noise
        };
        let mut extra_splits_len = context.args.extra_splits_len;
        let mut min_scene_len = context.args.min_scene_len;

        let mut zone_args_map = {
            let (remaining, parsed_args) =
                separated_list0(space1, parse_cli_arg).parse(zone_args).map_err(
                    |e: nom::Err<_>| anyhow!("Invalid zone override syntax: {}", e.to_string()),
                )?;
            if !remaining.trim().is_empty() {
                bail!("Unrecognized zone arguments: {}", remaining);
            }
            parsed_args.into_iter().collect::<HashMap<_, _>>()
        };

        if let Some(zone_passes) = zone_args_map.remove("--passes") {
            passes = zone_passes.unwrap().parse()?;
        }
        if [Encoder::aom, Encoder::vpx].contains(&encoder) && zone_args_map.contains_key("--rt") {
            passes = 1;
        }
        if let Some(zone_photon_noise) = zone_args_map.remove("--photon-noise") {
            photon_noise = Some(zone_photon_noise.unwrap().parse()?);
        }
        if let Some(zone_xs) =
            zone_args_map.remove("-x").or_else(|| zone_args_map.remove("--extra-split"))
        {
            extra_splits_len = Some(zone_xs.unwrap().parse()?);
        }
        if let Some(zone_min_scene_len) = zone_args_map.remove("--min-scene-len") {
            min_scene_len = zone_min_scene_len.unwrap().parse()?;
        }

        let raw_zone_args = if [Encoder::aom, Encoder::vpx].contains(&encoder) {
            zone_args_map
                .into_iter()
                .map(|(key, value)| {
                    value.map_or_else(|| key.to_string(), |value| format!("{key}={value}"))
                })
                .collect::<Vec<String>>()
        } else {
            zone_args_map
                .into_iter()
                .flat_map(|(key, value)| {
                    let mut vec = vec![key.to_string()];
                    if let Some(v) = value {
                        vec.push(v.to_string());
                    }
                    vec
                })
                .collect::<Vec<String>>()
        };

        // --- 验证参数 (逻辑不变) ---
        if !context.args.force {
            let help_text = {
                let [cmd, arg] = encoder.help_command();
                String::from_utf8(Command::new(cmd).arg(arg).output().unwrap().stdout).unwrap()
            };
            let valid_params = valid_params(&help_text, encoder);
            let interleaved_args: Vec<&str> = raw_zone_args
                .iter()
                .filter_map(|param| {
                    if param.starts_with('-') && [Encoder::aom, Encoder::vpx].contains(&encoder) {
                        param.split('=').next()
                    } else {
                        None
                    }
                })
                .collect();
            let invalid_params = invalid_params(&interleaved_args, &valid_params);

            for wrong_param in &invalid_params {
                eprintln!("'{wrong_param}' isn't a valid parameter for {encoder}");
                if let Some(suggestion) = suggest_fix(wrong_param, &valid_params) {
                    eprintln!("\tDid you mean '{suggestion}'?");
                }
            }

            if !invalid_params.is_empty() {
                println!("\nTo continue anyway, run av1an with '--force'");
                exit(1);
            }
        }

        // --- 合并参数 (逻辑不变) ---
        for arg in raw_zone_args {
            if arg.starts_with("--")
                || (arg.starts_with('-') && arg.chars().nth(1).is_some_and(char::is_alphabetic))
            {
                let key = arg.split_once('=').map_or(arg.as_str(), |split| split.0);
                if let Some(pos) = video_params
                    .iter()
                    .position(|param| param == key || param.starts_with(&format!("{key}=")))
                {
                    video_params.remove(pos);
                    if let Some(next) = video_params.get(pos) {
                        if !([Encoder::aom, Encoder::vpx].contains(&encoder)
                            || next.starts_with("--")
                            || (next.starts_with('-')
                                && next.chars().nth(1).is_some_and(char::is_alphabetic)))
                        {
                            video_params.remove(pos);
                        }
                    }
                }
            }
            video_params.push(arg);
        }

        Ok(Self {
            start_frame,
            end_frame,
            zone_overrides: Some(ZoneOptions {
                encoder,
                passes,
                video_params,
                photon_noise,
                extra_splits_len,
                min_scene_len,
            }),
        })
    }
}
