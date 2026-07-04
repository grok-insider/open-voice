//! OpenAI adapter: `/v1/audio/transcriptions` (STT) and `/v1/audio/speech`
//! (TTS). Also works against OpenAI-compatible bases (vLLM & friends) via
//! `base_url`.
//!
//! STT notes: only `whisper-1` supports `verbose_json` with word/segment
//! timestamps; `gpt-4o-transcribe*` return plain `json`. The adapter switches
//! response parsing on the model name. Batch uploads are capped at 25 MB and
//! OGG/Opus is not in the documented input list — the engine transcodes
//! first when needed.

use async_trait::async_trait;
use ov_core::capabilities::ProviderCapabilities;
use ov_core::domain::{
    AudioCodec, AudioOutput, Segment, SpeechRequest, TranscribeRequest, Transcript, Word,
};
use ov_core::ports::{BatchSpeechSynthesizer, BatchTranscriber, Provider};
use ov_core::{CoreError, CoreResult, ProviderId};
use serde::Deserialize;

use crate::http;

pub const MAX_UPLOAD_BYTES: u64 = 25 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct OpenAiSettings {
    pub base_url: String,
    pub stt_model: String,
    pub tts_model: String,
    pub tts_voice: String,
}

impl Default for OpenAiSettings {
    fn default() -> Self {
        OpenAiSettings {
            base_url: "https://api.openai.com".to_string(),
            stt_model: "whisper-1".to_string(),
            tts_model: "gpt-4o-mini-tts".to_string(),
            tts_voice: "marin".to_string(),
        }
    }
}

pub struct OpenAiProvider {
    client: reqwest::Client,
    api_key: String,
    settings: OpenAiSettings,
}

impl OpenAiProvider {
    pub fn new(api_key: impl Into<String>, settings: OpenAiSettings) -> Self {
        OpenAiProvider {
            client: http::client(),
            api_key: api_key.into(),
            settings,
        }
    }
}

#[derive(Debug, Deserialize)]
struct VerboseResponse {
    text: String,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    duration: Option<f64>,
    #[serde(default)]
    segments: Vec<VerboseSegment>,
    #[serde(default)]
    words: Vec<VerboseWord>,
}

#[derive(Debug, Deserialize)]
struct VerboseSegment {
    start: f64,
    end: f64,
    text: String,
}

#[derive(Debug, Deserialize)]
struct VerboseWord {
    word: String,
    start: f64,
    end: f64,
}

impl Provider for OpenAiProvider {
    fn id(&self) -> ProviderId {
        ProviderId::Openai
    }

    fn capabilities(&self) -> ProviderCapabilities {
        let whisper = self.settings.stt_model.starts_with("whisper");
        ProviderCapabilities {
            batch_stt: true,
            batch_tts: true,
            word_timestamps: whisper,
            segment_timestamps: whisper,
            prompt: true,
            max_upload_bytes: Some(MAX_UPLOAD_BYTES),
            stt_input_extensions: ["flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            ..Default::default()
        }
    }
}

#[async_trait]
impl BatchTranscriber for OpenAiProvider {
    async fn transcribe(&self, request: TranscribeRequest) -> CoreResult<Transcript> {
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.settings.stt_model.clone());
        let verbose = model.starts_with("whisper");
        let (data, file_name, mime) = http::source_parts(&request.source).await?;

        let file = reqwest::multipart::Part::bytes(data)
            .file_name(file_name)
            .mime_str(&mime)
            .map_err(|e| CoreError::InvalidInput(format!("invalid mime: {e}")))?;
        let mut form = reqwest::multipart::Form::new()
            .part("file", file)
            .text("model", model.clone());
        if let Some(language) = &request.language {
            form = form.text("language", language.primary());
        }
        if let Some(prompt) = &request.prompt {
            form = form.text("prompt", prompt.clone());
        }
        if verbose {
            form = form
                .text("response_format", "verbose_json")
                .text("timestamp_granularities[]", "word")
                .text("timestamp_granularities[]", "segment");
        } else {
            form = form.text("response_format", "json");
        }

        let response = self
            .client
            .post(format!(
                "{}/v1/audio/transcriptions",
                self.settings.base_url
            ))
            .bearer_auth(&self.api_key)
            .multipart(form)
            .send()
            .await
            .map_err(http::network_err)?;
        if !response.status().is_success() {
            return Err(http::error_for("openai", response).await);
        }

        let body: VerboseResponse = response.json().await.map_err(|e| CoreError::Provider {
            provider: "openai".into(),
            message: format!("decoding response: {e}"),
        })?;

        let mut transcript = Transcript::new(ProviderId::Openai, body.text);
        transcript.model = Some(model);
        transcript.language = body.language;
        transcript.duration = body.duration;
        transcript.segments = body
            .segments
            .into_iter()
            .map(|s| Segment::new(s.start, s.end, s.text.trim()))
            .collect();
        transcript.words = body
            .words
            .into_iter()
            .map(|w| Word::new(w.word, w.start, w.end))
            .collect();
        Ok(transcript)
    }
}

fn response_format(codec: AudioCodec) -> CoreResult<&'static str> {
    Ok(match codec {
        AudioCodec::Mp3 => "mp3",
        AudioCodec::Opus => "opus",
        AudioCodec::Aac => "aac",
        AudioCodec::Flac => "flac",
        AudioCodec::Wav => "wav",
        AudioCodec::Pcm => "pcm",
        AudioCodec::Mulaw | AudioCodec::Alaw => {
            return Err(CoreError::Unsupported {
                provider: "openai".into(),
                message: format!("output codec '{}'", codec.as_str()),
            })
        }
    })
}

fn mime_for(codec: AudioCodec) -> &'static str {
    match codec {
        AudioCodec::Mp3 => "audio/mpeg",
        AudioCodec::Opus => "audio/opus",
        AudioCodec::Aac => "audio/aac",
        AudioCodec::Flac => "audio/flac",
        AudioCodec::Wav => "audio/wav",
        AudioCodec::Pcm => "audio/pcm",
        AudioCodec::Mulaw => "audio/basic",
        AudioCodec::Alaw => "audio/alaw",
    }
}

#[async_trait]
impl BatchSpeechSynthesizer for OpenAiProvider {
    async fn synthesize(&self, request: SpeechRequest) -> CoreResult<AudioOutput> {
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.settings.tts_model.clone());
        let voice = request
            .voice
            .clone()
            .unwrap_or_else(|| self.settings.tts_voice.clone());
        let mut body = serde_json::json!({
            "model": model,
            "input": request.text,
            "voice": voice,
            "response_format": response_format(request.codec)?,
        });
        if let Some(speed) = request.speed {
            body["speed"] = serde_json::json!(speed);
        }
        if let Some(instructions) = &request.instructions {
            body["instructions"] = serde_json::json!(instructions);
        }

        let response = self
            .client
            .post(format!("{}/v1/audio/speech", self.settings.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(http::network_err)?;
        if !response.status().is_success() {
            return Err(http::error_for("openai", response).await);
        }
        let bytes = response.bytes().await.map_err(http::network_err)?;
        Ok(AudioOutput {
            bytes: bytes.to_vec(),
            mime: mime_for(request.codec).to_string(),
            codec: request.codec,
            provider: ProviderId::Openai,
            duration: None,
            metadata: None,
        })
    }
}
