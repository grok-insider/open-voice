//! ElevenLabs adapter: Scribe STT (`/v1/speech-to-text`) and TTS
//! (`/v1/text-to-speech/{voice_id}`). Auth is the `xi-api-key` header.
//!
//! STT returns a word list where entries are typed `word` / `spacing` /
//! `audio_event`; only `word` entries become domain `Word`s. Segments are
//! synthesized from word timings by `ov_core::domain::segments_from_words`.

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

/// Documented cap for direct uploads (5.0 GB).
pub const MAX_UPLOAD_BYTES: u64 = 5 * 1024 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct ElevenLabsSettings {
    pub base_url: String,
    pub stt_model: String,
    pub tts_model: String,
    pub tts_voice: String,
}

impl Default for ElevenLabsSettings {
    fn default() -> Self {
        ElevenLabsSettings {
            base_url: "https://api.elevenlabs.io".to_string(),
            stt_model: "scribe_v2".to_string(),
            tts_model: "eleven_multilingual_v2".to_string(),
            // "Rachel", the stock ElevenLabs voice.
            tts_voice: "21m00Tcm4TlvDq8ikWAM".to_string(),
        }
    }
}

pub struct ElevenLabsProvider {
    client: reqwest::Client,
    api_key: String,
    settings: ElevenLabsSettings,
}

impl ElevenLabsProvider {
    pub fn new(api_key: impl Into<String>, settings: ElevenLabsSettings) -> Self {
        ElevenLabsProvider {
            client: http::client(),
            api_key: api_key.into(),
            settings,
        }
    }
}

#[derive(Debug, Deserialize)]
struct SttResponse {
    #[serde(default)]
    language_code: Option<String>,
    text: String,
    #[serde(default)]
    words: Vec<SttWord>,
}

#[derive(Debug, Deserialize)]
struct SttWord {
    text: String,
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    start: Option<f64>,
    #[serde(default)]
    end: Option<f64>,
    #[serde(default)]
    speaker_id: Option<String>,
}

impl Provider for ElevenLabsProvider {
    fn id(&self) -> ProviderId {
        ProviderId::Elevenlabs
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            batch_stt: true,
            batch_tts: true,
            word_timestamps: true,
            segment_timestamps: true,
            diarization: true,
            multichannel: true,
            keyterms: true,
            max_upload_bytes: Some(MAX_UPLOAD_BYTES),
            // "All major audio and video formats are supported."
            stt_input_extensions: Vec::new(),
            ..Default::default()
        }
    }
}

#[async_trait]
impl BatchTranscriber for ElevenLabsProvider {
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
            .text("model_id", model.clone())
            .part("file", file)
            .text("timestamps_granularity", "word")
            .text("diarize", if request.diarize { "true" } else { "false" });
        if let Some(language) = &request.language {
            form = form.text("language_code", language.primary());
        }
        for term in &request.keyterms {
            form = form.text("keyterms", term.clone());
        }

        let response = self
            .client
            .post(format!("{}/v1/speech-to-text", self.settings.base_url))
            .header("xi-api-key", &self.api_key)
            .multipart(form)
            .send()
            .await
            .map_err(http::network_err)?;
        if !response.status().is_success() {
            return Err(http::error_for("elevenlabs", response).await);
        }

        let body: SttResponse = response.json().await.map_err(|e| CoreError::Provider {
            provider: "elevenlabs".into(),
            message: format!("decoding response: {e}"),
        })?;

        let mut transcript = Transcript::new(ProviderId::Elevenlabs, body.text);
        transcript.model = Some(model);
        transcript.language = body.language_code;
        transcript.words = body
            .words
            .into_iter()
            .filter(|w| w.kind.as_deref().unwrap_or("word") == "word")
            .filter_map(|w| {
                let (start, end) = (w.start?, w.end?);
                let mut word = Word::new(w.text, start, end);
                word.speaker = w.speaker_id;
                Some(word)
            })
            .collect();
        transcript.segments = segments_from_words(&transcript.words);
        transcript.duration = transcript.words.last().map(|w| w.end);
        Ok(transcript)
    }
}

fn output_format(codec: AudioCodec, sample_rate: Option<u32>) -> CoreResult<String> {
    match codec {
        AudioCodec::Mp3 => Ok(format!("mp3_{}_128", sample_rate.unwrap_or(44_100))),
        AudioCodec::Pcm => Ok(format!("pcm_{}", sample_rate.unwrap_or(24_000))),
        AudioCodec::Opus => Ok(format!("opus_{}_128", sample_rate.unwrap_or(48_000))),
        other => Err(CoreError::Unsupported {
            provider: "elevenlabs".into(),
            message: format!("output codec '{}' (use mp3, pcm, or opus)", other.as_str()),
        }),
    }
}

#[async_trait]
impl BatchSpeechSynthesizer for ElevenLabsProvider {
    async fn synthesize(&self, request: SpeechRequest) -> CoreResult<AudioOutput> {
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.settings.tts_model.clone());
        let voice = request
            .voice
            .clone()
            .unwrap_or_else(|| self.settings.tts_voice.clone());
        let format = output_format(request.codec, request.sample_rate)?;

        let mut body = serde_json::json!({
            "text": request.text,
            "model_id": model,
        });
        if let Some(language) = &request.language {
            // multilingual_v2 rejects language_code; only send it for models
            // that support enforcement (flash/turbo v2.5+).
            if !model.contains("multilingual_v2") {
                body["language_code"] = serde_json::json!(language.primary());
            }
        }
        if let Some(speed) = request.speed {
            body["voice_settings"] = serde_json::json!({ "speed": speed });
        }

        let response = self
            .client
            .post(format!(
                "{}/v1/text-to-speech/{voice}",
                self.settings.base_url
            ))
            .query(&[("output_format", format.as_str())])
            .header("xi-api-key", &self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(http::network_err)?;
        if !response.status().is_success() {
            return Err(http::error_for("elevenlabs", response).await);
        }
        let bytes = response.bytes().await.map_err(http::network_err)?;
        let mime = match request.codec {
            AudioCodec::Mp3 => "audio/mpeg",
            AudioCodec::Pcm => "audio/pcm",
            AudioCodec::Opus => "audio/opus",
            _ => "application/octet-stream",
        };
        Ok(AudioOutput {
            bytes: bytes.to_vec(),
            mime: mime.to_string(),
            codec: request.codec,
            provider: ProviderId::Elevenlabs,
            duration: None,
        })
    }
}
