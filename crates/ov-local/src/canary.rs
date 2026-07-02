//! Local Canary 1B v2 adapter (feature `canary`): NVIDIA's multilingual STT
//! model, int8 ONNX export, run in-process via `transcribe-rs`/ONNX Runtime.
//! No Python, no PyTorch, no network at inference time.
//!
//! Inference is synchronous and heavy, so it runs on a blocking thread. The
//! model is (re)loaded per call — acceptable for a batch CLI, and it keeps
//! this adapter stateless.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use ov_core::capabilities::ProviderCapabilities;
use ov_core::domain::{AudioSource, Segment, TranscribeRequest, Transcript};
use ov_core::ports::{AudioDecoder, AudioSpec, BatchTranscriber, Provider};
use ov_core::{CoreError, CoreResult, ProviderId};
use transcribe_rs::onnx::canary::{CanaryModel, CanaryParams};
use transcribe_rs::onnx::Quantization;

pub struct CanaryLocalProvider {
    model_dir: PathBuf,
    decoder: Arc<dyn AudioDecoder>,
}

impl CanaryLocalProvider {
    pub fn new(model_dir: PathBuf, decoder: Arc<dyn AudioDecoder>) -> Self {
        CanaryLocalProvider { model_dir, decoder }
    }

    fn run_inference(
        model_dir: &Path,
        wav_path: &Path,
        language: Option<String>,
    ) -> CoreResult<transcribe_rs::TranscriptionResult> {
        let mut model =
            CanaryModel::load(model_dir, &Quantization::Int8).map_err(|e| CoreError::Provider {
                provider: "local-canary".into(),
                message: format!("loading model: {e}"),
            })?;
        let samples = transcribe_rs::audio::read_wav_samples(wav_path)
            .map_err(|e| CoreError::Audio(format!("reading wav: {e}")))?;
        let params = CanaryParams {
            language,
            ..Default::default()
        };
        model
            .transcribe_with(&samples, &params)
            .map_err(|e| CoreError::Provider {
                provider: "local-canary".into(),
                message: format!("inference: {e}"),
            })
    }
}

impl Provider for CanaryLocalProvider {
    fn id(&self) -> ProviderId {
        ProviderId::LocalCanary
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            batch_stt: true,
            segment_timestamps: true,
            // Everything ffmpeg can decode is accepted (we transcode locally).
            stt_input_extensions: Vec::new(),
            ..Default::default()
        }
    }
}

#[async_trait]
impl BatchTranscriber for CanaryLocalProvider {
    async fn transcribe(&self, request: TranscribeRequest) -> CoreResult<Transcript> {
        let input = match &request.source {
            AudioSource::File(path) => path.clone(),
            AudioSource::Bytes { .. } => {
                return Err(CoreError::Unsupported {
                    provider: "local-canary".into(),
                    message: "in-memory sources are not supported; pass a file".into(),
                })
            }
        };

        // Canary consumes 16 kHz mono 16-bit WAV.
        let wav = self
            .decoder
            .decode_to_wav(&input, AudioSpec::STT_16K_MONO)
            .await?;
        let duration = self.decoder.probe_duration(&wav).await.unwrap_or(None);

        let model_dir = self.model_dir.clone();
        let language = request.language.as_ref().map(|l| l.primary());
        let language_for_transcript = language.clone();
        let wav_for_task = wav.clone();
        let result = tokio::task::spawn_blocking(move || {
            Self::run_inference(&model_dir, &wav_for_task, language)
        })
        .await
        .map_err(|e| CoreError::Provider {
            provider: "local-canary".into(),
            message: format!("inference task: {e}"),
        });
        std::fs::remove_file(&wav).ok();
        let result = result??;

        let mut transcript = Transcript::new(ProviderId::LocalCanary, result.text.trim());
        transcript.model = Some("canary-1b-v2".to_string());
        transcript.language = language_for_transcript;
        transcript.duration = duration;
        if let Some(segments) = result.segments {
            transcript.segments = segments
                .into_iter()
                .filter(|s| !s.text.trim().is_empty())
                .map(|s| Segment::new(f64::from(s.start), f64::from(s.end), s.text.trim()))
                .collect();
        }
        Ok(transcript)
    }
}
