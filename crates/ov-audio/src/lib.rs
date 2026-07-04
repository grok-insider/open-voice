//! # ov-audio
//!
//! Audio decode/probe adapter built on the `ffmpeg` binary. ffmpeg is the
//! pragmatic choice for v0.1: it decodes everything (WhatsApp OGG/Opus
//! included) and is wrapped onto PATH by the Nix package. A pure-Rust
//! `symphonia` adapter can implement the same `AudioDecoder` port later
//! without touching any consumer.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use async_trait::async_trait;
use ov_core::domain::AudioCodec;
use ov_core::ports::{AudioDecoder, AudioEncoder, AudioSpec};
use ov_core::{CoreError, CoreResult};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

/// MIME type guess from a file extension (used for multipart uploads).
pub fn mime_for_extension(ext: &str) -> &'static str {
    match ext.to_ascii_lowercase().as_str() {
        "mp3" | "mpga" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" | "oga" => "audio/ogg",
        "opus" => "audio/opus",
        "flac" => "audio/flac",
        "m4a" => "audio/mp4",
        "mp4" => "video/mp4",
        "aac" => "audio/aac",
        "webm" => "audio/webm",
        "mkv" => "video/x-matroska",
        "mpeg" => "video/mpeg",
        _ => "application/octet-stream",
    }
}

#[derive(Debug, Clone)]
pub struct FfmpegDecoder {
    ffmpeg: String,
    ffprobe: String,
}

impl Default for FfmpegDecoder {
    fn default() -> Self {
        FfmpegDecoder {
            ffmpeg: "ffmpeg".to_string(),
            ffprobe: "ffprobe".to_string(),
        }
    }
}

impl FfmpegDecoder {
    pub fn new(ffmpeg: impl Into<String>, ffprobe: impl Into<String>) -> Self {
        FfmpegDecoder {
            ffmpeg: ffmpeg.into(),
            ffprobe: ffprobe.into(),
        }
    }

    pub fn is_available(&self) -> bool {
        std::process::Command::new(&self.ffmpeg)
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    async fn run_ffmpeg(&self, args: &[&str]) -> CoreResult<Vec<u8>> {
        let output = Command::new(&self.ffmpeg)
            .args(["-y", "-hide_banner", "-loglevel", "error"])
            .args(args)
            .stdin(Stdio::null())
            .output()
            .await
            .map_err(|e| CoreError::Audio(format!("spawning {}: {e}", self.ffmpeg)))?;
        if !output.status.success() {
            return Err(CoreError::Audio(format!(
                "ffmpeg failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        Ok(output.stdout)
    }
}

#[async_trait]
impl AudioDecoder for FfmpegDecoder {
    async fn decode_to_wav(&self, input: &Path, spec: AudioSpec) -> CoreResult<PathBuf> {
        let out = tempfile::Builder::new()
            .prefix("openvoice.")
            .suffix(".wav")
            .tempfile()
            .map_err(|e| CoreError::Io(format!("creating temp wav: {e}")))?;
        // Keep the file on disk after the handle drops; callers clean up.
        let (_, path) = out
            .keep()
            .map_err(|e| CoreError::Io(format!("persisting temp wav: {e}")))?;
        let path_str = path.to_string_lossy().into_owned();
        let input_str = input.to_string_lossy().into_owned();
        let rate = spec.sample_rate.to_string();
        let channels = spec.channels.to_string();
        self.run_ffmpeg(&[
            "-i", &input_str, "-ac", &channels, "-ar", &rate, "-vn", "-f", "wav", &path_str,
        ])
        .await?;
        Ok(path)
    }

    async fn decode_to_pcm(&self, input: &Path, spec: AudioSpec) -> CoreResult<Vec<u8>> {
        let input_str = input.to_string_lossy().into_owned();
        let rate = spec.sample_rate.to_string();
        let channels = spec.channels.to_string();
        self.run_ffmpeg(&[
            "-i", &input_str, "-ac", &channels, "-ar", &rate, "-vn", "-f", "s16le", "pipe:1",
        ])
        .await
    }

    async fn probe_duration(&self, input: &Path) -> CoreResult<Option<f64>> {
        let output = Command::new(&self.ffprobe)
            .args([
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
            ])
            .arg(input)
            .stdin(Stdio::null())
            .output()
            .await
            .map_err(|e| CoreError::Audio(format!("spawning {}: {e}", self.ffprobe)))?;
        if !output.status.success() {
            return Ok(None);
        }
        let text = String::from_utf8_lossy(&output.stdout);
        Ok(text.trim().parse::<f64>().ok())
    }
}

#[async_trait]
impl AudioEncoder for FfmpegDecoder {
    async fn encode_wav(
        &self,
        wav: &[u8],
        codec: AudioCodec,
        sample_rate: Option<u32>,
    ) -> CoreResult<Vec<u8>> {
        // WAV in, WAV out, no resample: nothing to do.
        if codec == AudioCodec::Wav && sample_rate.is_none() {
            return Ok(wav.to_vec());
        }
        let format = match codec {
            AudioCodec::Wav => "wav",
            AudioCodec::Mp3 => "mp3",
            AudioCodec::Flac => "flac",
            AudioCodec::Aac => "adts",
            AudioCodec::Opus => "opus",
            AudioCodec::Pcm => "s16le",
            AudioCodec::Mulaw => "mulaw",
            AudioCodec::Alaw => "alaw",
        };
        let mut args: Vec<String> = vec![
            "-y".into(),
            "-hide_banner".into(),
            "-loglevel".into(),
            "error".into(),
            "-i".into(),
            "pipe:0".into(),
        ];
        if let Some(rate) = sample_rate {
            args.push("-ar".into());
            args.push(rate.to_string());
        }
        args.push("-f".into());
        args.push(format.into());
        args.push("pipe:1".into());

        let mut child = Command::new(&self.ffmpeg)
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| CoreError::Audio(format!("spawning {}: {e}", self.ffmpeg)))?;
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| CoreError::Audio("ffmpeg stdin unavailable".into()))?;
        let input = wav.to_vec();
        let writer = tokio::spawn(async move {
            let _ = stdin.write_all(&input).await;
            let _ = stdin.shutdown().await;
        });
        let output = child
            .wait_with_output()
            .await
            .map_err(|e| CoreError::Audio(format!("running ffmpeg: {e}")))?;
        writer.abort();
        if !output.status.success() {
            return Err(CoreError::Audio(format!(
                "ffmpeg encode failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        Ok(output.stdout)
    }
}

/// Write a minimal 16-bit PCM WAV file (test/tooling helper).
pub fn write_wav(path: &Path, sample_rate: u32, channels: u16, samples: &[i16]) -> CoreResult<()> {
    let data_len = (samples.len() * 2) as u32;
    let byte_rate = sample_rate * u32::from(channels) * 2;
    let block_align = channels * 2;
    let mut bytes = Vec::with_capacity(44 + samples.len() * 2);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&block_align.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    std::fs::write(path, bytes).map_err(|e| CoreError::Io(format!("writing wav: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wav(dir: &Path, sample_rate: u32, seconds: f64) -> PathBuf {
        let n = (sample_rate as f64 * seconds) as usize;
        let samples: Vec<i16> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                ((t * 440.0 * std::f64::consts::TAU).sin() * 12000.0) as i16
            })
            .collect();
        let path = dir.join("tone.wav");
        write_wav(&path, sample_rate, 1, &samples).unwrap();
        path
    }

    fn ffmpeg_or_skip() -> Option<FfmpegDecoder> {
        let decoder = FfmpegDecoder::default();
        if decoder.is_available() {
            Some(decoder)
        } else {
            eprintln!("skipping: ffmpeg not on PATH");
            None
        }
    }

    #[test]
    fn mime_guesses() {
        assert_eq!(mime_for_extension("OGG"), "audio/ogg");
        assert_eq!(mime_for_extension("mp3"), "audio/mpeg");
        assert_eq!(mime_for_extension("zzz"), "application/octet-stream");
    }

    #[tokio::test]
    async fn decodes_to_16k_mono_wav() {
        let Some(decoder) = ffmpeg_or_skip() else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let input = sine_wav(dir.path(), 44_100, 0.25);
        let out = decoder
            .decode_to_wav(&input, AudioSpec::STT_16K_MONO)
            .await
            .unwrap();
        let bytes = std::fs::read(&out).unwrap();
        assert_eq!(&bytes[0..4], b"RIFF");
        // Sample rate field at offset 24.
        let rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
        assert_eq!(rate, 16_000);
        let channels = u16::from_le_bytes([bytes[22], bytes[23]]);
        assert_eq!(channels, 1);
        std::fs::remove_file(out).ok();
    }

    #[tokio::test]
    async fn decodes_to_raw_pcm() {
        let Some(decoder) = ffmpeg_or_skip() else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let input = sine_wav(dir.path(), 16_000, 0.5);
        let pcm = decoder
            .decode_to_pcm(&input, AudioSpec::STT_16K_MONO)
            .await
            .unwrap();
        // 0.5s at 16kHz mono s16le = 16000 bytes; allow ffmpeg padding slack.
        assert!(pcm.len() >= 15_000, "pcm too short: {}", pcm.len());
    }

    #[tokio::test]
    async fn encodes_wav_to_mp3() {
        let Some(decoder) = ffmpeg_or_skip() else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let input = sine_wav(dir.path(), 24_000, 0.5);
        let wav = std::fs::read(&input).unwrap();
        let mp3 = decoder
            .encode_wav(&wav, AudioCodec::Mp3, None)
            .await
            .unwrap();
        assert!(mp3.len() > 500, "mp3 too small: {}", mp3.len());
        assert_ne!(&mp3[0..4], b"RIFF");
        // WAV passthrough returns the input unchanged.
        let same = decoder
            .encode_wav(&wav, AudioCodec::Wav, None)
            .await
            .unwrap();
        assert_eq!(same, wav);
    }

    #[tokio::test]
    async fn probes_duration() {
        let Some(decoder) = ffmpeg_or_skip() else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let input = sine_wav(dir.path(), 16_000, 1.0);
        let duration = decoder.probe_duration(&input).await.unwrap();
        let d = duration.expect("duration");
        assert!((d - 1.0).abs() < 0.1, "duration {d}");
    }
}
