//! Cartesia adapter: batch STT (`/stt`, `ink-whisper`) and TTS
//! (`/tts/bytes`, `sonic`). Every request carries the mandatory
//! `Cartesia-Version` header.
//!
//! Cartesia TTS has no default voice — a voice id must come from the request
//! or settings, otherwise the adapter fails fast with `NotConfigured`.

use async_trait::async_trait;
use ov_core::capabilities::ProviderCapabilities;
use ov_core::domain::{
    segments_from_words, AudioCodec, AudioOutput, SpeechRequest, TranscribeRequest, Transcript,
    Word,
};
use ov_core::ports::{BatchSpeechSynthesizer, BatchTranscriber, Provider};
use ov_core::{CoreError, CoreResult, ProviderId};
use serde::Deserialize;

use crate::http;

#[derive(Debug, Clone)]
pub struct CartesiaSettings {
    pub base_url: String,
    /// `Cartesia-Version` header value.
    pub version: String,
    pub stt_model: String,
    pub tts_model: String,
    /// Cartesia requires an explicit voice id; empty = unset.
    pub tts_voice: String,
}

impl Default for CartesiaSettings {
    fn default() -> Self {
        CartesiaSettings {
            base_url: "https://api.cartesia.ai".to_string(),
            version: "2026-03-01".to_string(),
            stt_model: "ink-whisper".to_string(),
            tts_model: "sonic-3.5".to_string(),
            tts_voice: String::new(),
        }
    }
}

pub struct CartesiaProvider {
    client: reqwest::Client,
    api_key: String,
    settings: CartesiaSettings,
}

impl CartesiaProvider {
    pub fn new(api_key: impl Into<String>, settings: CartesiaSettings) -> Self {
        CartesiaProvider {
            client: http::client(),
            api_key: api_key.into(),
            settings,
        }
    }
}

#[derive(Debug, Deserialize)]
struct SttResponse {
    text: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    duration: Option<f64>,
    #[serde(default)]
    words: Vec<SttWord>,
}

#[derive(Debug, Deserialize)]
struct SttWord {
    word: String,
    start: f64,
    end: f64,
}

impl Provider for CartesiaProvider {
    fn id(&self) -> ProviderId {
        ProviderId::Cartesia
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            batch_stt: true,
            batch_tts: true,
            word_timestamps: true,
            segment_timestamps: true,
            stt_input_extensions: [
                "flac", "m4a", "mp3", "mp4", "mpeg", "mpga", "oga", "ogg", "wav", "webm",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            ..Default::default()
        }
    }
}

#[async_trait]
impl BatchTranscriber for CartesiaProvider {
    async fn transcribe(&self, request: TranscribeRequest) -> CoreResult<Transcript> {
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.settings.stt_model.clone());
        let (data, file_name, mime) = http::source_parts(&request.source).await?;

        let file = reqwest::multipart::Part::bytes(data)
            .file_name(file_name)
            .mime_str(&mime)
            .map_err(|e| CoreError::InvalidInput(format!("invalid mime: {e}")))?;
        let mut form = reqwest::multipart::Form::new()
            .part("file", file)
            .text("model", model.clone())
            .text("timestamp_granularities[]", "word");
        if let Some(language) = &request.language {
            form = form.text("language", language.primary());
        }

        let response = self
            .client
            .post(format!("{}/stt", self.settings.base_url))
            .bearer_auth(&self.api_key)
            .header("Cartesia-Version", &self.settings.version)
            .multipart(form)
            .send()
            .await
            .map_err(http::network_err)?;
        if !response.status().is_success() {
            return Err(http::error_for("cartesia", response).await);
        }

        let body: SttResponse = response.json().await.map_err(|e| CoreError::Provider {
            provider: "cartesia".into(),
            message: format!("decoding response: {e}"),
        })?;

        let mut transcript = Transcript::new(ProviderId::Cartesia, body.text);
        transcript.model = Some(model);
        transcript.language = body.language;
        transcript.duration = body.duration;
        transcript.words = body
            .words
            .into_iter()
            .map(|w| Word::new(w.word, w.start, w.end))
            .collect();
        transcript.segments = segments_from_words(&transcript.words);
        Ok(transcript)
    }
}

fn output_format(codec: AudioCodec, sample_rate: Option<u32>) -> CoreResult<serde_json::Value> {
    match codec {
        AudioCodec::Mp3 => Ok(serde_json::json!({
            "container": "mp3",
            "sample_rate": sample_rate.unwrap_or(44_100),
            "bit_rate": 128_000,
        })),
        AudioCodec::Wav => Ok(serde_json::json!({
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": sample_rate.unwrap_or(44_100),
        })),
        AudioCodec::Pcm => Ok(serde_json::json!({
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": sample_rate.unwrap_or(24_000),
        })),
        other => Err(CoreError::Unsupported {
            provider: "cartesia".into(),
            message: format!("output codec '{}' (use mp3, wav, or pcm)", other.as_str()),
        }),
    }
}

#[async_trait]
impl BatchSpeechSynthesizer for CartesiaProvider {
    async fn synthesize(&self, request: SpeechRequest) -> CoreResult<AudioOutput> {
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.settings.tts_model.clone());
        let voice = request
            .voice
            .clone()
            .filter(|v| !v.is_empty())
            .or_else(|| Some(self.settings.tts_voice.clone()).filter(|v| !v.is_empty()))
            .ok_or_else(|| CoreError::NotConfigured {
                provider: "cartesia".into(),
                message: "no voice id (pass --voice or set providers.cartesia.tts_voice)".into(),
            })?;

        let mut body = serde_json::json!({
            "model_id": model,
            "transcript": request.text,
            "voice": { "mode": "id", "id": voice },
            "output_format": output_format(request.codec, request.sample_rate)?,
        });
        if let Some(language) = &request.language {
            body["language"] = serde_json::json!(language.primary());
        }
        if let Some(speed) = request.speed {
            body["generation_config"] = serde_json::json!({ "speed": speed });
        }

        let response = self
            .client
            .post(format!("{}/tts/bytes", self.settings.base_url))
            .bearer_auth(&self.api_key)
            .header("Cartesia-Version", &self.settings.version)
            .json(&body)
            .send()
            .await
            .map_err(http::network_err)?;
        if !response.status().is_success() {
            return Err(http::error_for("cartesia", response).await);
        }
        let bytes = response.bytes().await.map_err(http::network_err)?;
        let mime = match request.codec {
            AudioCodec::Mp3 => "audio/mpeg",
            AudioCodec::Wav => "audio/wav",
            AudioCodec::Pcm => "audio/pcm",
            _ => "application/octet-stream",
        };
        Ok(AudioOutput {
            bytes: bytes.to_vec(),
            mime: mime.to_string(),
            codec: request.codec,
            provider: ProviderId::Cartesia,
            duration: None,
            metadata: None,
        })
    }
}
