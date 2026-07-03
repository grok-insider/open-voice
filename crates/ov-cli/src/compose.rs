//! Composition root: builds the `Engine` from `Config`, registering every
//! configured adapter in `auto` preference order.
//!
//! STT order: local-canary → xai → elevenlabs → cartesia → openai.
//! TTS order: xai → elevenlabs → cartesia → openai.

use std::sync::Arc;

use ov_audio::FfmpegDecoder;
use ov_config::Config;
use ov_core::ProviderId;
use ov_engine::Engine;
use ov_providers::{
    CartesiaProvider, CartesiaSettings, ElevenLabsProvider, ElevenLabsSettings, OpenAiProvider,
    OpenAiSettings, XaiProvider, XaiSettings,
};

pub struct Composition {
    pub engine: Engine,
    pub decoder: Arc<FfmpegDecoder>,
    /// Which providers are configured (API key present / model installed).
    pub configured: Vec<ProviderId>,
}

fn xai(config: &Config, key: String) -> Arc<XaiProvider> {
    Arc::new(XaiProvider::new(
        key,
        XaiSettings {
            base_url: config.providers.xai.base_url.clone(),
            ws_url: config.providers.xai.ws_url.clone(),
            tts_voice: config.providers.xai.tts_voice.clone(),
        },
    ))
}

fn elevenlabs(config: &Config, key: String) -> Arc<ElevenLabsProvider> {
    Arc::new(ElevenLabsProvider::new(
        key,
        ElevenLabsSettings {
            base_url: config.providers.elevenlabs.base_url.clone(),
            stt_model: config.providers.elevenlabs.stt_model.clone(),
            tts_model: config.providers.elevenlabs.tts_model.clone(),
            tts_voice: config.providers.elevenlabs.tts_voice.clone(),
        },
    ))
}

fn cartesia(config: &Config, key: String) -> Arc<CartesiaProvider> {
    Arc::new(CartesiaProvider::new(
        key,
        CartesiaSettings {
            base_url: config.providers.cartesia.base_url.clone(),
            version: config.providers.cartesia.version.clone(),
            stt_model: config.providers.cartesia.stt_model.clone(),
            tts_model: config.providers.cartesia.tts_model.clone(),
            tts_voice: config.providers.cartesia.tts_voice.clone(),
        },
    ))
}

fn openai(config: &Config, key: String) -> Arc<OpenAiProvider> {
    Arc::new(OpenAiProvider::new(
        key,
        OpenAiSettings {
            base_url: config.providers.openai.base_url.clone(),
            stt_model: config.providers.openai.stt_model.clone(),
            tts_model: config.providers.openai.tts_model.clone(),
            tts_voice: config.providers.openai.tts_voice.clone(),
        },
    ))
}

/// Whether the local Canary model is installed (feature-independent check so
/// `providers list` can explain how to enable it).
pub fn local_canary_installed(config: &Config) -> bool {
    ov_local::models::CANARY_1B_V2.is_installed(&config.models_dir())
}

pub fn local_canary_compiled() -> bool {
    cfg!(feature = "local")
}

pub fn local_qwen3_compiled() -> bool {
    cfg!(feature = "local-tts")
}

/// Where the local Qwen3-TTS model would load from, if anywhere:
/// explicit config dir → open-voice models dir → shared HF cache.
pub fn local_qwen3_source(config: &Config) -> Option<Qwen3Source> {
    let explicit = config.local.tts_model_dir.trim();
    if !explicit.is_empty() {
        return Some(Qwen3Source::Dir(std::path::PathBuf::from(explicit)));
    }
    let spec = &ov_local::models::QWEN3_TTS_CUSTOM_VOICE;
    if spec.is_installed(&config.models_dir()) {
        return Some(Qwen3Source::Dir(spec.dir(&config.models_dir())));
    }
    if ov_local::models::hf_cache_present(spec.repo) {
        return Some(Qwen3Source::HfCache);
    }
    None
}

pub enum Qwen3Source {
    Dir(std::path::PathBuf),
    HfCache,
}

pub fn build(config: &Config) -> Composition {
    let decoder = Arc::new(FfmpegDecoder::default());
    let mut builder = Engine::builder().decoder(decoder.clone());
    let mut configured = Vec::new();

    // Local Canary first: free, private, and best-tested for Spanish.
    #[cfg(feature = "local")]
    if local_canary_installed(config) {
        let provider = Arc::new(ov_local::CanaryLocalProvider::new(
            ov_local::models::CANARY_1B_V2.dir(&config.models_dir()),
            decoder.clone(),
        ));
        builder = builder.stt(provider);
        configured.push(ProviderId::LocalCanary);
    }

    // Local Qwen3-TTS first for TTS: free, private, quality parity with the
    // hosted voices for the supported languages.
    #[cfg(feature = "local-tts")]
    if let Some(source) = local_qwen3_source(config) {
        let model_dir = match source {
            Qwen3Source::Dir(dir) => Some(dir),
            Qwen3Source::HfCache => None,
        };
        let provider = Arc::new(ov_local::Qwen3TtsLocalProvider::new(
            model_dir,
            Some(config.local.tts_voice.clone()),
            decoder.clone(),
        ));
        builder = builder.tts(provider);
        configured.push(ProviderId::LocalQwen3);
    }

    if let Some(key) = config.api_key(ProviderId::Xai) {
        let provider = xai(config, key);
        builder = builder
            .stt(provider.clone())
            .tts(provider.clone())
            .streaming_stt(provider.clone())
            .streaming_tts(provider);
        configured.push(ProviderId::Xai);
    }
    if let Some(key) = config.api_key(ProviderId::Elevenlabs) {
        let provider = elevenlabs(config, key);
        builder = builder.stt(provider.clone()).tts(provider);
        configured.push(ProviderId::Elevenlabs);
    }
    if let Some(key) = config.api_key(ProviderId::Cartesia) {
        let provider = cartesia(config, key);
        builder = builder.stt(provider.clone()).tts(provider);
        configured.push(ProviderId::Cartesia);
    }
    if let Some(key) = config.api_key(ProviderId::Openai) {
        let provider = openai(config, key);
        builder = builder.stt(provider.clone()).tts(provider);
        configured.push(ProviderId::Openai);
    }

    Composition {
        engine: builder.build(),
        decoder,
        configured,
    }
}
