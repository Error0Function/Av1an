use clap::ValueEnum;
use ffmpeg::format::Pixel;
use serde::{Deserialize, Serialize};
use splines::{Interpolation, Key, Spline};
use std::{
    cmp,
    cmp::Ordering,
    collections::HashSet,
    fmt::{self, Write},
    path::PathBuf,
    thread::available_parallelism,
};
use tracing::{debug, trace};

use crate::{
    broker::EncoderCrash, chunk::Chunk, progress_bar::update_mp_msg, vmaf::read_weighted_vmaf,
    Encoder, ProbingStatistic,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetQuality {
    pub vmaf_res: String,
    pub vmaf_scaler: String,
    pub vmaf_filter: Option<String>,
    pub vmaf_threads: usize,
    pub model: Option<String>,
    pub probing_rate: usize,
    pub probing_speed: Option<u8>,
    pub probes: u32,
    pub target: f64,
    pub min_q: u32,
    pub max_q: u32,
    pub encoder: Encoder,
    pub pix_format: Pixel,
    pub temp: String,
    pub workers: usize,
    pub video_params: Vec<String>,
    pub vspipe_args: Vec<String>,
    pub probe_slow: bool,
    pub probing_vmaf_features: Vec<VmafFeature>,
    pub probing_statistic: ProbingStatistic,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionMethod {
    ForcedAnchor,     // 策略0: 首次探测的固定锚点
    RegulaFalsi,      // 策略1：割线法/Falsi算法 (用于内插)
    RdModel,          // 策略2: 率失真模型 (用于外插)
    SplineCatmullRom, // 策略3: Catmull-Rom样条插值 (备用)
    SplineLinear,     // 策略3: 线性样条插值 (备用)
    BinarySearch,     // 策略4: 二分查找 (备用)
}

// 为 PredictionMethod 实现 Display trait，使其能被优雅地打印
impl fmt::Display for PredictionMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PredictionMethod::ForcedAnchor => write!(f, "Forced Anchor"),
            PredictionMethod::RegulaFalsi => write!(f, "Regula Falsi (Secant)"),
            PredictionMethod::RdModel => write!(f, "Rate-Distortion Model"),
            PredictionMethod::SplineCatmullRom => write!(f, "Spline (Catmull-Rom)"),
            PredictionMethod::SplineLinear => write!(f, "Spline (Linear)"),
            PredictionMethod::BinarySearch => write!(f, "Binary Search"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ValueEnum)]
pub enum VmafFeature {
    #[value(name = "default")]
    Default,
    #[value(name = "weighted")]
    Weighted,
    #[value(name = "neg")]
    Neg,
    #[value(name = "motionless")]
    Motionless,
    #[value(name = "uhd")]
    Uhd,
}

impl TargetQuality {
    /// 为单个视频块（Chunk）寻找最优量化参数（Quantizer）的核心函数。
    /// 支持从中断处恢复探测过程。
    ///
    /// # 目标
    /// 找到一个 `q` 值，使其编码后的视频VMAF分数尽可能精确地匹配用户设定的 `target` VMAF。
    ///
    /// # 策略
    /// 这是一个迭代搜索过程，结合了多种策略以实现高效和稳健：
    /// 1. **启动策略**: 第一次探测使用固定的高质量 `q=1` 作为锚点。
    /// 2. **智能预测**: 当数据点足够时，优先使用基于率失真理论的模型进行预测。
    /// 3. **分层回退**: 如果高级模型失效，会自动降级到样条插值，最终到最简单的二分查找。
    /// 4. **安全机制**: 整个过程包含了对数值溢出、下溢和模型不稳定的全面防御。
    ///
    /// # 参数
    /// - `chunk`: 一个可变的 Chunk 引用，函数会读取并更新其 `probe_history`。
    /// - `worker_id`: 用于在日志中标识工作线程。
    /// - `on_probe_complete`: 一个回调函数，在每次探测完成后被调用，用于持久化中间状态。
    pub fn per_shot_target_quality<F>(
        &self,
        chunk: &mut Chunk,
        worker_id: Option<usize>,
        mut on_probe_complete: F,
    ) -> anyhow::Result<u32>
    where
        F: FnMut(&Chunk),
    {
        // 修复借用检查：提前获取不可变的数据
        let chunk_name = chunk.name();

        if chunk.probe_history.is_none() {
            chunk.probe_history = Some(Vec::new());
        }

        // 我们将 history 的可变借用限制在循环内部，以释放对 chunk 的锁定
        let mut lower_quantizer_limit = self.min_q;
        let mut upper_quantizer_limit = self.max_q;
        if let Some(history) = &chunk.probe_history {
            for (q, _, score) in history.iter() {
                if *score > self.target {
                    lower_quantizer_limit = (*q + 1).min(upper_quantizer_limit);
                } else {
                    upper_quantizer_limit = q.saturating_sub(1).max(lower_quantizer_limit);
                }
            }
        }

        let initial_probe_count = chunk.probe_history.as_ref().unwrap().len();
        if initial_probe_count > 0 {
            debug!(
                "Chunk {}: Resuming quality search with {} existing probes. New Q range [{}, {}]",
                chunk_name, initial_probe_count, lower_quantizer_limit, upper_quantizer_limit
            );
        } else {
            debug!(
                "Chunk {}: Starting quality search for target VMAF {:.2} within Q range [{}, {}]",
                chunk_name, self.target, lower_quantizer_limit, upper_quantizer_limit
            );
        }

        let stop_reason = loop {
            // 在每次循环开始时，获取 history 的可变引用
            let history = chunk.probe_history.as_mut().unwrap();

            let quantizer_score_history: Vec<(u32, f64)> =
                history.iter().map(|(q, _, s)| (*q, *s)).collect();

            if history.len() >= self.probes as usize {
                break SkipProbingReason::ProbeLimitReached;
            }

            let (next_quantizer, prediction_method) = predict_quantizer(
                lower_quantizer_limit,
                upper_quantizer_limit,
                &quantizer_score_history,
                self.target,
            );

            if quantizer_score_history.iter().any(|(q, _)| *q == next_quantizer) {
                break SkipProbingReason::None;
            }

            // 在这里释放可变借用，以便在 vmaf_probe 中不可变地借用 chunk
            let _ = history;

            if let Some(worker_id) = worker_id {
                update_mp_msg(
                    worker_id,
                    format!(
                        "Targeting Quality {:.2} - Testing Q={}",
                        self.target, next_quantizer
                    ),
                );
            }
            let probe_path = self.vmaf_probe(chunk, next_quantizer as usize)?;
            let score =
                read_weighted_vmaf(&probe_path, self.probing_statistic.clone()).map_err(|e| {
                    Box::new(EncoderCrash {
                        exit_status: std::process::ExitStatus::default(),
                        source_pipe_stderr: String::new().into(),
                        ffmpeg_pipe_stderr: None,
                        stderr: format!("VMAF calculation failed: {e}").into(),
                        stdout: String::new().into(),
                    })
                })?;

            // 重新获取可变借用以更新历史
            let history = chunk.probe_history.as_mut().unwrap();
            history.push((next_quantizer, prediction_method, score));

            // 释放可变借用，以便在回调中不可变地借用 chunk
            let _ = history;
            on_probe_complete(chunk);

            let history_len = chunk.probe_history.as_ref().unwrap().len();
            let score_within_tolerance = within_tolerance(score, self.target);
            const MINIMUM_PROBES: usize = 3;
            if (history_len >= MINIMUM_PROBES && score_within_tolerance)
                || history_len >= self.probes as usize
            {
                break if score_within_tolerance {
                    SkipProbingReason::WithinTolerance
                } else {
                    SkipProbingReason::ProbeLimitReached
                };
            }

            if score > self.target {
                lower_quantizer_limit = (next_quantizer + 1).min(upper_quantizer_limit);
            } else {
                upper_quantizer_limit = next_quantizer.saturating_sub(1).max(lower_quantizer_limit);
            }

            if lower_quantizer_limit > upper_quantizer_limit {
                break if score > self.target {
                    SkipProbingReason::QuantizerTooLow
                } else {
                    SkipProbingReason::QuantizerTooHigh
                };
            }
        };

        let history = chunk.probe_history.as_ref().unwrap();
        let (final_q, _, final_score) = *history
            .iter()
            .filter(|(_, _, score)| within_tolerance(*score, self.target))
            .max_by_key(|(q, _, _)| *q)
            .or_else(|| {
                history.iter().min_by(|(_, _, s1), (_, _, s2)| {
                    (s1 - self.target)
                        .abs()
                        .partial_cmp(&(s2 - self.target).abs())
                        .unwrap_or(Ordering::Equal)
                })
            })
            .unwrap();

        log_final_summary(
            &chunk_name,
            self.target,
            history,
            final_q,
            final_score,
            stop_reason,
        );

        Ok(final_q)
    }

    fn vmaf_probe(&self, chunk: &Chunk, q: usize) -> Result<PathBuf, Box<EncoderCrash>> {
        let vmaf_threads = if self.vmaf_threads == 0 {
            vmaf_auto_threads(self.workers)
        } else {
            self.vmaf_threads
        };

        let cmd = self.encoder.probe_cmd(
            self.temp.clone(),
            chunk.index,
            q,
            self.pix_format,
            self.probing_rate,
            self.probing_speed,
            vmaf_threads,
            self.video_params.clone(),
            self.probe_slow,
        );

        let future = async {
            let source_cmd = chunk.source_cmd.clone();
            let cmd = cmd.clone();

            tokio::task::spawn_blocking(move || {
                let mut source = if let [pipe_cmd, args @ ..] = &*source_cmd {
                    std::process::Command::new(pipe_cmd)
                        .args(args)
                        .stderr(std::process::Stdio::piped())
                        .stdout(std::process::Stdio::piped())
                        .spawn()
                        .map_err(|e| EncoderCrash {
                            exit_status: std::process::ExitStatus::default(),
                            source_pipe_stderr: format!("Failed to spawn source: {e}").into(),
                            ffmpeg_pipe_stderr: None,
                            stderr: String::new().into(),
                            stdout: String::new().into(),
                        })?
                } else {
                    unreachable!()
                };

                let source_stdout = source.stdout.take().unwrap();

                let mut source_pipe = if let [ffmpeg, args @ ..] = &*cmd.0 {
                    std::process::Command::new(ffmpeg)
                        .args(args)
                        .stdin(source_stdout)
                        .stdout(std::process::Stdio::piped())
                        .stderr(std::process::Stdio::piped())
                        .spawn()
                        .map_err(|e| EncoderCrash {
                            exit_status: std::process::ExitStatus::default(),
                            source_pipe_stderr: format!("Failed to spawn ffmpeg: {e}").into(),
                            ffmpeg_pipe_stderr: None,
                            stderr: String::new().into(),
                            stdout: String::new().into(),
                        })?
                } else {
                    unreachable!()
                };

                let source_pipe_stdout = source_pipe.stdout.take().unwrap();

                let mut enc_pipe = if let [cmd, args @ ..] = &*cmd.1 {
                    std::process::Command::new(cmd.as_ref())
                        .args(args.iter().map(AsRef::as_ref))
                        .stdin(source_pipe_stdout)
                        .stdout(std::process::Stdio::piped())
                        .stderr(std::process::Stdio::piped())
                        .spawn()
                        .map_err(|e| EncoderCrash {
                            exit_status: std::process::ExitStatus::default(),
                            source_pipe_stderr: String::new().into(),
                            ffmpeg_pipe_stderr: None,
                            stderr: format!("Failed to spawn encoder: {e}").into(),
                            stdout: String::new().into(),
                        })?
                } else {
                    unreachable!()
                };

                // Drop stdout to prevent buffer deadlock
                drop(enc_pipe.stdout.take());

                // Start reading stderr concurrently to prevent deadlock
                use std::{io::Read, thread};

                let source_stderr = source.stderr.take().unwrap();
                let source_pipe_stderr = source_pipe.stderr.take().unwrap();
                let enc_stderr = enc_pipe.stderr.take().unwrap();

                let stderr_thread1 = thread::spawn(move || {
                    let mut buf = Vec::new();
                    let mut stderr = source_stderr;
                    stderr.read_to_end(&mut buf).ok();
                    buf
                });

                let stderr_thread2 = thread::spawn(move || {
                    let mut buf = Vec::new();
                    let mut stderr = source_pipe_stderr;
                    stderr.read_to_end(&mut buf).ok();
                    buf
                });

                let stderr_thread3 = thread::spawn(move || {
                    let mut buf = Vec::new();
                    let mut stderr = enc_stderr;
                    stderr.read_to_end(&mut buf).ok();
                    buf
                });

                // Wait for encoder & other processes to finish
                let enc_status = enc_pipe.wait().map_err(|e| EncoderCrash {
                    exit_status: std::process::ExitStatus::default(),
                    source_pipe_stderr: String::new().into(),
                    ffmpeg_pipe_stderr: None,
                    stderr: format!("Failed to wait for encoder: {e}").into(),
                    stdout: String::new().into(),
                })?;

                let _ = source_pipe.wait();
                let _ = source.wait();

                // Collect stderr after process finishes
                let stderr_handles = (
                    stderr_thread1.join().unwrap_or_default(),
                    stderr_thread2.join().unwrap_or_default(),
                    stderr_thread3.join().unwrap_or_default(),
                );

                if !enc_status.success() {
                    return Err(EncoderCrash {
                        exit_status: enc_status,
                        source_pipe_stderr: stderr_handles.0.into(),
                        ffmpeg_pipe_stderr: Some(stderr_handles.1.into()),
                        stderr: stderr_handles.2.into(),
                        stdout: String::new().into(),
                    });
                }

                Ok(())
            })
            .await
            .unwrap()
        };

        let rt = tokio::runtime::Builder::new_current_thread().enable_io().build().unwrap();
        rt.block_on(future)?;

        let extension = match self.encoder {
            crate::encoder::Encoder::x264 => "264",
            crate::encoder::Encoder::x265 => "hevc",
            _ => "ivf",
        };

        let probe_name = std::path::Path::new(&chunk.temp)
            .join("split")
            .join(format!("v_{index:05}_{q}.{extension}", index = chunk.index));
        let fl_path = std::path::Path::new(&chunk.temp)
            .join("split")
            .join(format!("{index}.json", index = chunk.index));

        let features: HashSet<_> = self.probing_vmaf_features.iter().copied().collect();
        let use_weighted = features.contains(&VmafFeature::Weighted);
        let use_neg = features.contains(&VmafFeature::Neg);
        let use_uhd = features.contains(&VmafFeature::Uhd);
        let disable_motion = features.contains(&VmafFeature::Motionless);

        let default_model = match (use_uhd, use_neg) {
            (true, true) => Some("vmaf_4k_v0.6.1neg.json"),
            (true, false) => Some("vmaf_4k_v0.6.1.json"),
            (false, true) => Some("vmaf_v0.6.1neg.json"),
            (false, false) => None,
        };

        let model: Option<&str> = self.model.as_deref().or(default_model);

        if use_weighted {
            crate::vmaf::run_vmaf_weighted(
                &probe_name,
                chunk.source_cmd.as_slice(),
                self.vspipe_args.clone(),
                &fl_path,
                model,
                &self.vmaf_res,
                &self.vmaf_scaler,
                self.probing_rate,
                self.vmaf_filter.as_deref(),
                self.vmaf_threads,
                chunk.frame_rate,
                disable_motion,
            )
            .map_err(|e| {
                Box::new(EncoderCrash {
                    exit_status: std::process::ExitStatus::default(),
                    source_pipe_stderr: String::new().into(),
                    ffmpeg_pipe_stderr: None,
                    stderr: format!("VMAF calculation failed: {e}").into(),
                    stdout: String::new().into(),
                })
            })?;
        } else {
            crate::vmaf::run_vmaf(
                &probe_name,
                chunk.source_cmd.as_slice(),
                self.vspipe_args.clone(),
                &fl_path,
                model,
                &self.vmaf_res,
                &self.vmaf_scaler,
                self.probing_rate,
                self.vmaf_filter.as_deref(),
                self.vmaf_threads,
                chunk.frame_rate,
                disable_motion,
            )?;
        }

        Ok(fl_path)
    }
}

/// 对一组点 (x, y) 执行简单的线性回归。
fn linear_regression(points: &[(f64, f64)]) -> Option<(f64, f64)> {
    let n = points.len() as f64;
    if n < 2.0 {
        return None;
    }

    let sum_x: f64 = points.iter().map(|(x, _)| *x).sum();
    let sum_y: f64 = points.iter().map(|(_, y)| *y).sum();
    let sum_xy: f64 = points.iter().map(|(x, y)| x * y).sum();
    let sum_xx: f64 = points.iter().map(|(x, _)| x * x).sum();

    let denominator = n * sum_xx - sum_x.powi(2);

    // 避免除以零，这种情况发生在所有x值都相同时
    if denominator.abs() < 1e-9 {
        return None;
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denominator;
    let intercept = (sum_y - slope * sum_x) / n;

    Some((slope, intercept))
}

/// 该函数实现了分层预测策略，确保在各种数据分布下的准确性和稳健性。
/// 新版本引入了 Regula Falsi 算法以加速收敛。
fn predict_quantizer(
    lower_quantizer_limit: u32,
    upper_quantizer_limit: u32,
    quantizer_score_history: &[(u32, f64)],
    target: f64,
) -> (u32, PredictionMethod) {
    // --- 策略0: 启发式启动 ---
    if quantizer_score_history.is_empty() {
        trace!("Probe 1: Using Forced Anchor strategy.");
        const FIRST_PROBE_Q: u32 = 1;
        return (
            FIRST_PROBE_Q.clamp(lower_quantizer_limit, upper_quantizer_limit),
            PredictionMethod::ForcedAnchor,
        );
    }

    if quantizer_score_history.len() < 2 {
        trace!("Probe 2: Not enough data for models, using Binary Search.");
        return (
            (lower_quantizer_limit + upper_quantizer_limit) / 2,
            PredictionMethod::BinarySearch,
        );
    }

    // --- 策略1: Regula Falsi (割线法变体，优先使用) ---
    // 寻找一个“包围”目标值的区间 (一个点高于目标，一个点低于目标)
    let mut point_low: Option<(f64, f64)> = None;
    let mut point_high: Option<(f64, f64)> = None;

    for &(q, score) in quantizer_score_history {
        let q_f64 = q as f64;
        if score < target {
            // 找到低于目标的点中，最接近目标的一个 (即分数最高的)
            if point_low.is_none() || score > point_low.unwrap().1 {
                point_low = Some((q_f64, score));
            }
        } else if score > target {
            // 找到高于目标的点中，最接近目标的一个 (即分数最低的)
            if point_high.is_none() || score < point_high.unwrap().1 {
                point_high = Some((q_f64, score));
            }
        } else {
            // 碰巧找到了精确匹配的点
            return (q, PredictionMethod::RegulaFalsi);
        }
    }

    if let (Some((q_low, vmaf_low)), Some((q_high, vmaf_high))) = (point_low, point_high) {
        // 我们有一个完美的包围区间，使用割线法预测
        let f_low = vmaf_low - target;
        let f_high = vmaf_high - target;

        if (f_high - f_low).abs() > 1e-9 {
            let predicted_q = q_low - f_low * (q_high - q_low) / (f_high - f_low);

            trace!("Prediction using Regula Falsi (Secant) Method.");
            let final_q = predicted_q.round() as u32;

            // 确保预测值在已知的最佳边界内。q_high 对应高分(低Q)，q_low 对应低分(高Q)。
            // 所以新Q值应在 (q_high, q_low) 之间。
            let upper_bound = (q_low - 1.0).max(q_high + 1.0).round() as u32;
            let lower_bound = (q_high + 1.0).min(q_low - 1.0).round() as u32;

            return (
                final_q.clamp(lower_bound, upper_bound),
                PredictionMethod::RegulaFalsi,
            );
        }
    }

    // --- 策略2: 率失真（RD）模型 (当所有点都在目标的一侧时使用) ---
    let rd_prediction = || -> Option<f64> {
        if quantizer_score_history.iter().any(|(q, _)| *q == 0) {
            return None;
        }
        let points: Vec<(f64, f64)> =
            quantizer_score_history.iter().map(|(q, s)| ((*q as f64).ln(), *s)).collect();
        if let Some((slope, intercept)) = linear_regression(&points) {
            if slope < -1e-9 {
                let predicted_q = ((target - intercept) / slope).exp();
                if predicted_q.is_finite() {
                    return Some(predicted_q);
                }
            }
        }
        None
    };

    // --- 分层回退逻辑 ---
    let (predicted_f64, method) = if let Some(q) = rd_prediction() {
        trace!("Prediction using Rate-Distortion Model (extrapolation).");
        (q, PredictionMethod::RdModel)
    } else {
        trace!("RD Model failed, falling back to Spline Interpolation.");
        let mut sorted_history = quantizer_score_history.to_vec();
        sorted_history.sort_by_key(|(q, _)| *q);

        let keys = sorted_history
            .iter()
            .map(|(q, s)| Key::new(*s, *q as f64, Interpolation::CatmullRom))
            .collect();
        if let Some(q) = Spline::from_vec(keys).sample(target) {
            (q, PredictionMethod::SplineCatmullRom)
        } else {
            let keys = sorted_history
                .iter()
                .map(|(q, s)| Key::new(*s, *q as f64, Interpolation::Linear))
                .collect();
            if let Some(q) = Spline::from_vec(keys).sample(target) {
                (q, PredictionMethod::SplineLinear)
            } else {
                trace!("All models failed, falling back to Binary Search.");
                (
                    (lower_quantizer_limit as f64 + upper_quantizer_limit as f64) / 2.0,
                    PredictionMethod::BinarySearch,
                )
            }
        }
    };

    // --- 最终的数值安全转换 ---
    if !predicted_f64.is_finite() {
        trace!("Predictor returned non-finite value, overriding with Binary Search.");
        return (
            (lower_quantizer_limit + upper_quantizer_limit) / 2,
            PredictionMethod::BinarySearch,
        );
    }

    let clamped_f64 =
        predicted_f64.clamp(lower_quantizer_limit as f64, upper_quantizer_limit as f64);
    let final_q = clamped_f64.round() as u32;

    (final_q, method)
}

/// 检查一个分数是否在目标的容忍度范围内。
/// 使用绝对容忍度，对于高分段目标更精确。
fn within_tolerance(score: f64, target: f64) -> bool {
    const ABSOLUTE_SCORE_TOLERANCE: f64 = 0.05; // VMAF分数误差必须小于 0.05
    (score - target).abs() < ABSOLUTE_SCORE_TOLERANCE
}

pub fn vmaf_auto_threads(workers: usize) -> usize {
    const OVER_PROVISION_FACTOR: f64 = 1.25;
    let threads = available_parallelism()
        .expect("Unrecoverable: Failed to get thread count")
        .get();
    cmp::max(
        ((threads / workers) as f64 * OVER_PROVISION_FACTOR) as usize,
        1,
    )
}

#[derive(Copy, Clone)]
pub enum SkipProbingReason {
    QuantizerTooHigh,
    QuantizerTooLow,
    WithinTolerance,
    ProbeLimitReached,
    None,
}

/// 在探测结束后，打印一个详细的总结报告。
pub fn log_final_summary(
    chunk_name: &str,
    target_vmaf: f64,
    history: &[(u32, PredictionMethod, f64)],
    final_q: u32,
    final_score: f64,
    stop_reason: SkipProbingReason,
) {
    use crate::progress_bar::println_above_progress_bar;

    // 创建一个可变字符串来构建整个报告
    let mut report = String::new();

    // 使用 writeln! 宏将所有内容写入字符串中
    // .unwrap() 在这里是安全的，因为向String写入不会失败
    writeln!(
        report,
        "\n=================================================="
    )
    .unwrap();
    writeln!(report, " Chunk Analysis Report: {}", chunk_name).unwrap();
    writeln!(report, " Target VMAF Quality: {:.3}", target_vmaf).unwrap();
    writeln!(report, "--------------------------------------------------").unwrap();
    writeln!(report, " Probing Path:").unwrap();
    for (i, (q, method, score)) in history.iter().enumerate() {
        writeln!(
            report,
            "  Probe {:>2}: Q={:<2} -> VMAF={:<7.3} (Predicted by: {})",
            i + 1,
            q,
            score,
            method
        )
        .unwrap();
    }
    writeln!(report, "--------------------------------------------------").unwrap();
    write!(report, " Stop Reason: ").unwrap();
    match stop_reason {
        SkipProbingReason::None => {
            writeln!(report, "Converged (a previously tested Q was predicted)").unwrap()
        },
        SkipProbingReason::QuantizerTooHigh => writeln!(
            report,
            "Early Exit (Quantizer range exhausted, score is too low)"
        )
        .unwrap(),
        SkipProbingReason::QuantizerTooLow => writeln!(
            report,
            "Early Exit (Quantizer range exhausted, score is too high)"
        )
        .unwrap(),
        SkipProbingReason::WithinTolerance => {
            writeln!(report, "Success (Score is within tolerance)").unwrap()
        },
        SkipProbingReason::ProbeLimitReached => {
            writeln!(report, "Stopped (Probe limit reached)").unwrap()
        },
    }
    writeln!(
        report,
        " Final Selection: Q={} (VMAF: {:.3})",
        final_q, final_score
    )
    .unwrap();
    writeln!(
        report,
        "==================================================\n"
    )
    .unwrap();

    // 使用新函数一次性、安全地打印整个报告
    println_above_progress_bar(&report);
}

#[inline]
pub const fn adapt_probing_rate(rate: usize) -> usize {
    match rate {
        1..=4 => rate,
        _ => 1,
    }
}
