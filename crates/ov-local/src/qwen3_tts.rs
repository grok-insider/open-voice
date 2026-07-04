//! Local Qwen3-TTS adapter (feature `qwen3-tts`): Alibaba's
//! Qwen3-TTS-12Hz-1.7B-CustomVoice run in-process via `any-tts`/Candle.
//! Named speakers (ryan, serena, vivian, ...), instruct-style control, 10
//! languages incl. en/es/ru. CPU works everywhere; the `qwen3-tts-cuda`
//! feature builds Candle's CUDA kernels for GPU inference.
//!
//! The engine natively emits 24 kHz WAV; other requested codecs are
//! re-encoded through the `AudioEncoder` port (ffmpeg).
//!
//! Model resolution order (mirrors any-tts's own tiers):
//! 1. an explicit local model dir (config `local.tts_model_dir` or the
//!    open-voice models dir when `openvoice models fetch qwen3-tts` ran),
//! 2. the shared Hugging Face cache / download by repo id.

use std::path::PathBuf;
use std::sync::Arc;

use any_tts::{load_model, ModelType, SynthesisRequest, TtsConfig};
use async_trait::async_trait;
use ov_core::capabilities::ProviderCapabilities;
use ov_core::domain::{AudioCodec, AudioOutput, Language, SpeechRequest};
use ov_core::ports::{AudioEncoder, BatchSpeechSynthesizer, Provider};
use ov_core::{CoreError, CoreResult, ProviderId};

pub const HF_REPO: &str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice";
pub const DEFAULT_VOICE: &str = "ryan";

/// Map an ISO 639-1 (or BCP-47) tag onto the language *names* the Qwen3
/// backend expects. Unknown tags pass through unchanged so model-native
/// names (or `auto`) keep working.
pub fn language_name(language: &Language) -> String {
    match language.primary().as_str() {
        "en" => "English".to_string(),
        "es" => "Spanish".to_string(),
        "ru" => "Russian".to_string(),
        "zh" => "Chinese".to_string(),
        "ja" => "Japanese".to_string(),
        "ko" => "Korean".to_string(),
        "de" => "German".to_string(),
        "fr" => "French".to_string(),
        "pt" => "Portuguese".to_string(),
        "it" => "Italian".to_string(),
        _ => language.code().to_string(),
    }
}

/// Qwen3 CustomVoice speaker ids are lowercase (`ryan`, not `Ryan`).
pub fn normalize_voice(voice: &str) -> String {
    voice.trim().to_ascii_lowercase()
}

pub struct Qwen3TtsLocalProvider {
    /// Local model directory; `None` resolves via HF cache/download.
    model_dir: Option<PathBuf>,
    default_voice: String,
    encoder: Arc<dyn AudioEncoder>,
}

impl Qwen3TtsLocalProvider {
    pub fn new(
        model_dir: Option<PathBuf>,
        default_voice: Option<String>,
        encoder: Arc<dyn AudioEncoder>,
    ) -> Self {
        Qwen3TtsLocalProvider {
            model_dir,
            default_voice: default_voice
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| DEFAULT_VOICE.to_string()),
            encoder,
        }
    }

    fn run_inference(
        model_dir: Option<PathBuf>,
        text: String,
        language: Option<String>,
        voice: String,
        instruct: Option<String>,
    ) -> CoreResult<Vec<u8>> {
        let mut config = TtsConfig::new(ModelType::Qwen3Tts);
        config = match model_dir {
            Some(dir) => config.with_model_path(dir.to_string_lossy().into_owned()),
            None => config.with_hf_model_id(HF_REPO),
        };
        let model = load_model(config).map_err(|e| CoreError::Provider {
            provider: "local-qwen3".into(),
            message: format!("loading model: {e}"),
        })?;

        let mut request = SynthesisRequest::new(&text).with_voice(&voice);
        if let Some(language) = &language {
            request = request.with_language(language);
        }
        if let Some(instruct) = &instruct {
            request = request.with_instruct(instruct);
        }
        let audio = model
            .synthesize(&request)
            .map_err(|e| CoreError::Provider {
                provider: "local-qwen3".into(),
                message: format!("synthesis: {e}"),
            })?;
        Ok(audio.get_wav())
    }
}

impl Provider for Qwen3TtsLocalProvider {
    fn id(&self) -> ProviderId {
        ProviderId::LocalQwen3
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            batch_tts: true,
            ..Default::default()
        }
    }
}

#[async_trait]
impl BatchSpeechSynthesizer for Qwen3TtsLocalProvider {
    async fn synthesize(&self, request: SpeechRequest) -> CoreResult<AudioOutput> {
        if request.text.trim().is_empty() {
            return Err(CoreError::InvalidInput("empty text".into()));
        }
        if request.speed.is_some() {
            // Qwen3 CustomVoice has no speed control; style comes from
            // `--instructions`. Warn instead of failing so `auto` chains
            // don't bounce off the local engine.
            tracing::warn!("local-qwen3 ignores --speed (use --instructions for style)");
        }

        let model_dir = self.model_dir.clone();
        let text = request.text.clone();
        let language = request.language.as_ref().map(language_name);
        let voice = normalize_voice(
            request
                .voice
                .as_deref()
                .filter(|v| !v.trim().is_empty())
                .unwrap_or(&self.default_voice),
        );
        let instruct = request.instructions.clone();

        let wav = tokio::task::spawn_blocking(move || {
            Self::run_inference(model_dir, text, language, voice, instruct)
        })
        .await
        .map_err(|e| CoreError::Provider {
            provider: "local-qwen3".into(),
            message: format!("inference task: {e}"),
        })??;

        let bytes = self
            .encoder
            .encode_wav(&wav, request.codec, request.sample_rate)
            .await?;
        let mime = match request.codec {
            AudioCodec::Wav => "audio/wav",
            AudioCodec::Mp3 => "audio/mpeg",
            AudioCodec::Flac => "audio/flac",
            AudioCodec::Aac => "audio/aac",
            AudioCodec::Opus => "audio/opus",
            AudioCodec::Pcm => "audio/pcm",
            AudioCodec::Mulaw => "audio/basic",
            AudioCodec::Alaw => "audio/alaw",
        };
        Ok(AudioOutput {
            bytes,
            mime: mime.to_string(),
            codec: request.codec,
            provider: ProviderId::LocalQwen3,
            duration: None,
            metadata: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_iso_tags_to_names() {
        assert_eq!(language_name(&Language::new("es")), "Spanish");
        assert_eq!(language_name(&Language::new("es-MX")), "Spanish");
        assert_eq!(language_name(&Language::new("ru")), "Russian");
        // Unknown tags pass through (e.g. already a name, or `auto`).
        assert_eq!(language_name(&Language::new("auto")), "auto");
    }

    #[test]
    fn voices_are_lowercased() {
        assert_eq!(normalize_voice("Ryan"), "ryan");
        assert_eq!(normalize_voice(" Uncle_Fu "), "uncle_fu");
    }
}
