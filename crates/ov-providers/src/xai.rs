//! xAI (Grok Voice) adapter. The most capable remote provider in v0.1:
//!
//! * Batch STT — `POST /v1/stt` (multipart; **the `file` field must be the
//!   last multipart field**, per the xAI docs).
//! * Batch TTS — `POST /v1/tts` (JSON in, audio bytes out).
//! * Streaming STT — `wss://.../v1/stt` (binary PCM frames in, JSON
//!   transcript events out).
//! * Streaming TTS — `wss://.../v1/tts` (`text.delta`/`text.done` in,
//!   base64 `audio.delta` out).

use async_trait::async_trait;
use base64::Engine as _;
use futures_util::{SinkExt, StreamExt};
use ov_core::capabilities::ProviderCapabilities;
use ov_core::domain::{
    AudioCodec, AudioOutput, AudioSource, Language, SpeechRequest, TranscribeRequest, Transcript,
    Word,
};
use ov_core::ports::{
    AudioEvent, AudioEventStream, BatchSpeechSynthesizer, BatchTranscriber, Provider,
    StreamTranscribeRequest, StreamingSpeechSynthesizer, StreamingTranscriber, TranscriptEvent,
    TranscriptStream,
};
use ov_core::{CoreError, CoreResult, ProviderId};
use serde::{Deserialize, Serialize};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;

use crate::http;

pub const MAX_UPLOAD_BYTES: u64 = 500 * 1024 * 1024;

pub const XAI_BUILT_IN_VOICES: &[(&str, &str, &str)] = &[
    ("eve", "Eve", "Energetic, upbeat"),
    ("ara", "Ara", "Warm, friendly"),
    ("rex", "Rex", "Confident, clear"),
    ("sal", "Sal", "Smooth, balanced"),
    ("leo", "Leo", "Authoritative, strong"),
];

#[derive(Debug, Clone)]
pub struct XaiSettings {
    pub base_url: String,
    /// WebSocket base, e.g. `wss://api.x.ai` (tests use `ws://127.0.0.1:...`).
    pub ws_url: String,
    pub tts_voice: String,
}

impl Default for XaiSettings {
    fn default() -> Self {
        XaiSettings {
            base_url: "https://api.x.ai".to_string(),
            ws_url: "wss://api.x.ai".to_string(),
            tts_voice: "eve".to_string(),
        }
    }
}

pub struct XaiProvider {
    client: reqwest::Client,
    api_key: String,
    settings: XaiSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XaiVoice {
    #[serde(alias = "id")]
    pub voice_id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub gender: Option<String>,
    #[serde(default)]
    pub accent: Option<String>,
    #[serde(default)]
    pub age: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub use_case: Option<String>,
    #[serde(default)]
    pub tone: Option<String>,
    #[serde(default)]
    pub created_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomVoiceList {
    #[serde(default)]
    pub voices: Vec<XaiVoice>,
    #[serde(default)]
    pub pagination_token: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CustomVoiceCreateRequest {
    pub file: AudioSource,
    pub name: String,
    pub description: Option<String>,
    pub gender: Option<String>,
    pub accent: Option<String>,
    pub age: Option<String>,
    pub language: Option<String>,
    pub use_case: Option<String>,
    pub tone: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct CustomVoiceUpdateRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gender: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub age: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_case: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tone: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RealtimeAgentRequest {
    pub text: String,
    pub model: Option<String>,
    pub instructions: Option<String>,
    pub voice: Option<String>,
    pub reasoning_effort: Option<String>,
    pub output_codec: AudioCodec,
    pub output_sample_rate: u32,
    pub input_sample_rate: u32,
    pub text_only: bool,
    pub manual_turn: bool,
    pub vad_threshold: Option<f32>,
    pub vad_silence_duration_ms: Option<u64>,
    pub vad_prefix_padding_ms: Option<u64>,
    pub transcription_model: Option<String>,
    pub language_hint: Option<String>,
}

impl RealtimeAgentRequest {
    pub fn text(text: impl Into<String>) -> Self {
        RealtimeAgentRequest {
            text: text.into(),
            model: None,
            instructions: None,
            voice: None,
            reasoning_effort: None,
            output_codec: AudioCodec::Pcm,
            output_sample_rate: 24_000,
            input_sample_rate: 24_000,
            text_only: false,
            manual_turn: false,
            vad_threshold: None,
            vad_silence_duration_ms: None,
            vad_prefix_padding_ms: None,
            transcription_model: None,
            language_hint: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeAgentTurn {
    pub text: String,
    pub audio: Vec<u8>,
    pub audio_mime: String,
    pub audio_codec: AudioCodec,
    pub output_sample_rate: u32,
    pub input_transcript: Option<String>,
    pub conversation_id: Option<String>,
    pub response_id: Option<String>,
    pub response_status: Option<String>,
}

impl XaiProvider {
    pub fn new(api_key: impl Into<String>, settings: XaiSettings) -> Self {
        XaiProvider {
            client: http::client(),
            api_key: api_key.into(),
            settings,
        }
    }

    fn ws_request(
        &self,
        url: &str,
    ) -> CoreResult<tokio_tungstenite::tungstenite::handshake::client::Request> {
        let mut request = url
            .into_client_request()
            .map_err(|e| CoreError::InvalidInput(format!("bad websocket url: {e}")))?;
        let value = format!("Bearer {}", self.api_key)
            .parse()
            .map_err(|_| CoreError::InvalidInput("api key is not a valid header value".into()))?;
        request.headers_mut().insert("Authorization", value);
        Ok(request)
    }

    pub async fn list_custom_voices(
        &self,
        limit: Option<u32>,
        pagination_token: Option<&str>,
    ) -> CoreResult<CustomVoiceList> {
        let mut request = self
            .client
            .get(format!("{}/v1/custom-voices", self.settings.base_url))
            .bearer_auth(&self.api_key);
        if let Some(limit) = limit {
            request = request.query(&[("limit", limit.to_string())]);
        }
        if let Some(token) = pagination_token {
            request = request.query(&[("pagination_token", token.to_string())]);
        }
        let response = request.send().await.map_err(http::network_err)?;
        if !response.status().is_success() {
            return Err(http::error_for("xai", response).await);
        }
        response.json().await.map_err(|e| CoreError::Provider {
            provider: "xai".into(),
            message: format!("decoding custom voices: {e}"),
        })
    }

    pub async fn get_custom_voice(&self, voice_id: &str) -> CoreResult<XaiVoice> {
        let response = self
            .client
            .get(format!(
                "{}/v1/custom-voices/{voice_id}",
                self.settings.base_url
            ))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(http::network_err)?;
        if !response.status().is_success() {
            return Err(http::error_for("xai", response).await);
        }
        response.json().await.map_err(|e| CoreError::Provider {
            provider: "xai".into(),
            message: format!("decoding custom voice: {e}"),
        })
    }

    pub async fn create_custom_voice(
        &self,
        request: CustomVoiceCreateRequest,
    ) -> CoreResult<XaiVoice> {
        let (data, file_name, mime) = http::source_parts(&request.file).await?;
        let file = reqwest::multipart::Part::bytes(data)
            .file_name(file_name)
            .mime_str(&mime)
            .map_err(|e| CoreError::InvalidInput(format!("invalid mime: {e}")))?;
        let mut form = reqwest::multipart::Form::new().text("name", request.name);
        form = text_field(form, "description", request.description);
        form = text_field(form, "gender", request.gender);
        form = text_field(form, "accent", request.accent);
        form = text_field(form, "age", request.age);
        form = text_field(form, "language", request.language);
        form = text_field(form, "use_case", request.use_case);
        form = text_field(form, "tone", request.tone);
        form = form.part("file", file);

        let response = self
            .client
            .post(format!("{}/v1/custom-voices", self.settings.base_url))
            .bearer_auth(&self.api_key)
            .multipart(form)
            .send()
            .await
            .map_err(http::network_err)?;
        if !response.status().is_success() {
            return Err(http::error_for("xai", response).await);
        }
        response.json().await.map_err(|e| CoreError::Provider {
            provider: "xai".into(),
            message: format!("decoding custom voice: {e}"),
        })
    }

    pub async fn update_custom_voice(
        &self,
        voice_id: &str,
        request: CustomVoiceUpdateRequest,
    ) -> CoreResult<XaiVoice> {
        let body = serde_json::to_value(request)
            .map_err(|e| CoreError::InvalidInput(format!("encoding update body: {e}")))?;
        let empty = body.as_object().map(|o| o.is_empty()).unwrap_or(false);
        if empty {
            return Err(CoreError::InvalidInput(
                "provide at least one field to update".into(),
            ));
        }
        let response = self
            .client
            .patch(format!(
                "{}/v1/custom-voices/{voice_id}",
                self.settings.base_url
            ))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(http::network_err)?;
        if !response.status().is_success() {
            return Err(http::error_for("xai", response).await);
        }
        response.json().await.map_err(|e| CoreError::Provider {
            provider: "xai".into(),
            message: format!("decoding custom voice: {e}"),
        })
    }

    pub async fn delete_custom_voice(&self, voice_id: &str) -> CoreResult<bool> {
        let response = self
            .client
            .delete(format!(
                "{}/v1/custom-voices/{voice_id}",
                self.settings.base_url
            ))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(http::network_err)?;
        if !response.status().is_success() {
            return Err(http::error_for("xai", response).await);
        }
        if response.status().as_u16() == 204 {
            return Ok(true);
        }
        let body = response.text().await.map_err(http::network_err)?;
        if body.trim().is_empty() {
            return Ok(true);
        }
        let value: serde_json::Value =
            serde_json::from_str(&body).map_err(|e| CoreError::Provider {
                provider: "xai".into(),
                message: format!("decoding delete response: {e}"),
            })?;
        Ok(value
            .get("deleted")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true))
    }

    pub async fn realtime_text_turn(
        &self,
        request: RealtimeAgentRequest,
    ) -> CoreResult<RealtimeAgentTurn> {
        let (output_mime, output_sample_rate) =
            realtime_audio_format(request.output_codec, Some(request.output_sample_rate))?;
        let (input_mime, input_sample_rate) =
            realtime_audio_format(AudioCodec::Pcm, Some(request.input_sample_rate))?;
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| "grok-voice-latest".to_string());
        let mut url = format!(
            "{}/v1/realtime?model={}",
            self.settings.ws_url,
            urlencoding::encode(&model)
        );
        if let Some(effort) = &request.reasoning_effort {
            url.push_str(&format!(
                "&reasoning.effort={}",
                urlencoding::encode(effort)
            ));
        }

        let ws_request = self.ws_request(&url)?;
        let (socket, _) = connect_async(ws_request)
            .await
            .map_err(|e| CoreError::Network(format!("websocket connect: {e}")))?;
        let (mut sink, mut stream) = socket.split();

        let voice = request
            .voice
            .clone()
            .unwrap_or_else(|| self.settings.tts_voice.clone());
        let mut session = serde_json::json!({
            "model": model,
            "voice": voice,
            "audio": {
                "input": { "format": { "type": input_mime, "rate": input_sample_rate } },
                "output": { "format": { "type": output_mime, "rate": output_sample_rate } }
            },
            "turn_detection": if request.manual_turn {
                serde_json::Value::Null
            } else {
                serde_json::json!({ "type": "server_vad" })
            }
        });
        if let Some(instructions) = &request.instructions {
            session["instructions"] = serde_json::json!(instructions);
        }
        if let Some(effort) = &request.reasoning_effort {
            session["reasoning"] = serde_json::json!({ "effort": effort });
        }
        if !request.manual_turn {
            if let Some(threshold) = request.vad_threshold {
                session["turn_detection"]["threshold"] = serde_json::json!(threshold);
            }
            if let Some(ms) = request.vad_silence_duration_ms {
                session["turn_detection"]["silence_duration_ms"] = serde_json::json!(ms);
            }
            if let Some(ms) = request.vad_prefix_padding_ms {
                session["turn_detection"]["prefix_padding_ms"] = serde_json::json!(ms);
            }
        }
        if request.transcription_model.is_some() || request.language_hint.is_some() {
            let mut transcription = serde_json::json!({
                "model": request
                    .transcription_model
                    .clone()
                    .unwrap_or_else(|| "grok-transcribe".to_string())
            });
            if let Some(language_hint) = &request.language_hint {
                transcription["language_hint"] = serde_json::json!(language_hint);
            }
            session["audio"]["input"]["transcription"] = transcription;
        }
        let session_update = serde_json::json!({
            "type": "session.update",
            "session": session,
        });
        sink.send(Message::Text(session_update.to_string()))
            .await
            .map_err(|e| CoreError::Network(format!("websocket send: {e}")))?;

        let item = serde_json::json!({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": request.text }]
            }
        });
        sink.send(Message::Text(item.to_string()))
            .await
            .map_err(|e| CoreError::Network(format!("websocket send: {e}")))?;
        let modalities = if request.text_only {
            serde_json::json!(["text"])
        } else {
            serde_json::json!(["audio", "text"])
        };
        let response_create = serde_json::json!({
            "type": "response.create",
            "response": { "modalities": modalities }
        });
        sink.send(Message::Text(response_create.to_string()))
            .await
            .map_err(|e| CoreError::Network(format!("websocket send: {e}")))?;

        let mut audio = Vec::new();
        let mut transcript_delta = String::new();
        let mut transcript_done = None;
        let mut text_delta = String::new();
        let mut input_transcript = None;
        let mut conversation_id = None;
        let mut response_id = None;
        let mut response_status = None;

        while let Some(message) = stream.next().await {
            let text = match message {
                Ok(Message::Text(text)) => text,
                Ok(Message::Close(_)) => break,
                Ok(_) => continue,
                Err(e) => return Err(CoreError::Network(format!("websocket: {e}"))),
            };
            let value: serde_json::Value =
                serde_json::from_str(&text).map_err(|e| CoreError::Provider {
                    provider: "xai".into(),
                    message: format!("decoding realtime event: {e}"),
                })?;
            let kind = value.get("type").and_then(serde_json::Value::as_str);
            match kind {
                Some("conversation.created") => {
                    conversation_id = value
                        .get("conversation")
                        .and_then(|v| v.get("id"))
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string);
                }
                Some("response.created") => {
                    response_id = value
                        .get("response")
                        .and_then(|v| v.get("id"))
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string);
                }
                Some("response.output_audio.delta") => {
                    if let Some(encoded) = value.get("delta").and_then(serde_json::Value::as_str) {
                        let bytes = base64::engine::general_purpose::STANDARD
                            .decode(encoded)
                            .map_err(|e| CoreError::Provider {
                                provider: "xai".into(),
                                message: format!("decoding realtime audio: {e}"),
                            })?;
                        audio.extend_from_slice(&bytes);
                    }
                }
                Some("response.output_audio_transcript.delta") => {
                    if let Some(delta) = value.get("delta").and_then(serde_json::Value::as_str) {
                        transcript_delta.push_str(delta);
                    }
                }
                Some("response.output_audio_transcript.done") => {
                    transcript_done = value
                        .get("transcript")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string);
                }
                Some("response.text.delta") | Some("response.output_text.delta") => {
                    if let Some(delta) = value.get("delta").and_then(serde_json::Value::as_str) {
                        text_delta.push_str(delta);
                    }
                }
                Some("conversation.item.input_audio_transcription.updated")
                | Some("conversation.item.input_audio_transcription.completed") => {
                    input_transcript = value
                        .get("transcript")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string);
                }
                Some("response.done") => {
                    response_id = value
                        .get("response")
                        .and_then(|v| v.get("id"))
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string)
                        .or(response_id);
                    response_status = value
                        .get("response")
                        .and_then(|v| v.get("status"))
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string);
                    break;
                }
                Some("error") => return Err(realtime_error(value)),
                _ => {}
            }
        }

        Ok(RealtimeAgentTurn {
            text: transcript_done.unwrap_or_else(|| {
                if transcript_delta.is_empty() {
                    text_delta
                } else {
                    transcript_delta
                }
            }),
            audio,
            audio_mime: output_mime,
            audio_codec: request.output_codec,
            output_sample_rate,
            input_transcript,
            conversation_id,
            response_id,
            response_status,
        })
    }
}

fn realtime_audio_format(codec: AudioCodec, sample_rate: Option<u32>) -> CoreResult<(String, u32)> {
    let name = match codec {
        AudioCodec::Pcm => "audio/pcm",
        AudioCodec::Mulaw => "audio/pcmu",
        AudioCodec::Alaw => "audio/pcma",
        other => {
            return Err(CoreError::Unsupported {
                provider: "xai".into(),
                message: format!(
                    "realtime codec '{}' (use pcm, mulaw, or alaw)",
                    other.as_str()
                ),
            })
        }
    };
    Ok((name.to_string(), sample_rate.unwrap_or(24_000)))
}

fn realtime_error(value: serde_json::Value) -> CoreError {
    let message = value
        .get("error")
        .and_then(|v| v.get("message"))
        .and_then(serde_json::Value::as_str)
        .or_else(|| value.get("message").and_then(serde_json::Value::as_str))
        .unwrap_or("realtime ws error")
        .to_string();
    CoreError::Provider {
        provider: "xai".into(),
        message,
    }
}

fn text_field(
    form: reqwest::multipart::Form,
    name: &'static str,
    value: Option<String>,
) -> reqwest::multipart::Form {
    match value.filter(|v| !v.trim().is_empty()) {
        Some(value) => form.text(name, value),
        None => form,
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
    text: String,
    start: f64,
    end: f64,
    #[serde(default)]
    speaker: Option<u32>,
}

fn map_words(words: Vec<SttWord>) -> Vec<Word> {
    words
        .into_iter()
        .map(|w| {
            let mut word = Word::new(w.text, w.start, w.end);
            word.speaker = w.speaker.map(|s| format!("speaker_{s}"));
            word
        })
        .collect()
}

impl Provider for XaiProvider {
    fn id(&self) -> ProviderId {
        ProviderId::Xai
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            batch_stt: true,
            batch_tts: true,
            streaming_stt: true,
            streaming_tts: true,
            word_timestamps: true,
            segment_timestamps: true,
            diarization: true,
            multichannel: true,
            keyterms: true,
            max_upload_bytes: Some(MAX_UPLOAD_BYTES),
            stt_input_extensions: [
                "wav", "mp3", "ogg", "opus", "flac", "aac", "mp4", "m4a", "mkv",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            ..Default::default()
        }
    }
}

#[async_trait]
impl BatchTranscriber for XaiProvider {
    async fn transcribe(&self, request: TranscribeRequest) -> CoreResult<Transcript> {
        let (data, file_name, mime) = http::source_parts(&request.source).await?;

        // Field order matters: xAI requires `file` to be the LAST field.
        let mut form = reqwest::multipart::Form::new();
        if let Some(language) = &request.language {
            form = form
                .text("language", language.primary())
                .text("format", "true");
        }
        if request.diarize {
            form = form.text("diarize", "true");
        }
        for term in &request.keyterms {
            form = form.text("keyterm", term.clone());
        }
        let file = reqwest::multipart::Part::bytes(data)
            .file_name(file_name)
            .mime_str(&mime)
            .map_err(|e| CoreError::InvalidInput(format!("invalid mime: {e}")))?;
        form = form.part("file", file);

        let response = self
            .client
            .post(format!("{}/v1/stt", self.settings.base_url))
            .bearer_auth(&self.api_key)
            .multipart(form)
            .send()
            .await
            .map_err(http::network_err)?;
        if !response.status().is_success() {
            return Err(http::error_for("xai", response).await);
        }

        let body: SttResponse = response.json().await.map_err(|e| CoreError::Provider {
            provider: "xai".into(),
            message: format!("decoding response: {e}"),
        })?;

        let mut transcript = Transcript::new(ProviderId::Xai, body.text);
        transcript.model = Some("grok-stt".to_string());
        transcript.language = body.language;
        transcript.duration = body.duration;
        transcript.words = map_words(body.words);
        transcript.segments = ov_core::domain::segments_from_words(&transcript.words);
        Ok(transcript)
    }
}

fn codec_fields(codec: AudioCodec, sample_rate: Option<u32>) -> CoreResult<(String, u32)> {
    let name = match codec {
        AudioCodec::Mp3 => "mp3",
        AudioCodec::Wav => "wav",
        AudioCodec::Pcm => "pcm",
        AudioCodec::Mulaw => "mulaw",
        AudioCodec::Alaw => "alaw",
        other => {
            return Err(CoreError::Unsupported {
                provider: "xai".into(),
                message: format!(
                    "output codec '{}' (use mp3, wav, pcm, mulaw, or alaw)",
                    other.as_str()
                ),
            })
        }
    };
    Ok((name.to_string(), sample_rate.unwrap_or(24_000)))
}

fn mime_for_codec(codec: AudioCodec) -> &'static str {
    match codec {
        AudioCodec::Mp3 => "audio/mpeg",
        AudioCodec::Wav => "audio/wav",
        AudioCodec::Pcm => "audio/pcm",
        AudioCodec::Mulaw => "audio/basic",
        AudioCodec::Alaw => "audio/alaw",
        AudioCodec::Opus => "audio/opus",
        AudioCodec::Flac => "audio/flac",
        AudioCodec::Aac => "audio/aac",
    }
}

fn redact_audio_fields(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            for key in ["audio", "audio_base64", "data"] {
                if map.get(key).and_then(serde_json::Value::as_str).is_some() {
                    map.insert(key.to_string(), serde_json::json!("<base64 audio omitted>"));
                }
            }
            for value in map.values_mut() {
                redact_audio_fields(value);
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                redact_audio_fields(value);
            }
        }
        _ => {}
    }
}

fn audio_base64(value: &serde_json::Value) -> Option<&str> {
    value
        .get("audio")
        .or_else(|| value.get("audio_base64"))
        .or_else(|| value.get("data"))
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            value
                .get("data")
                .and_then(|v| v.get("audio"))
                .and_then(serde_json::Value::as_str)
        })
}

#[async_trait]
impl BatchSpeechSynthesizer for XaiProvider {
    async fn synthesize(&self, request: SpeechRequest) -> CoreResult<AudioOutput> {
        let voice = request
            .voice
            .clone()
            .unwrap_or_else(|| self.settings.tts_voice.clone());
        let language = request
            .language
            .as_ref()
            .map(Language::code)
            .unwrap_or("auto")
            .to_string();
        let (codec, sample_rate) = codec_fields(request.codec, request.sample_rate)?;
        let mut output_format = serde_json::json!({ "codec": codec, "sample_rate": sample_rate });
        if let Some(bit_rate) = request.bit_rate {
            output_format["bit_rate"] = serde_json::json!(bit_rate);
        }
        let mut body = serde_json::json!({
            "text": request.text,
            "language": language,
            "voice_id": voice,
            "output_format": output_format,
        });
        if let Some(model) = &request.model {
            body["model"] = serde_json::json!(model);
        }
        if let Some(speed) = request.speed {
            body["speed"] = serde_json::json!(speed);
        }
        if let Some(latency) = request.optimize_streaming_latency {
            body["optimize_streaming_latency"] = serde_json::json!(latency);
        }
        if let Some(text_normalization) = request.text_normalization {
            body["text_normalization"] = serde_json::json!(text_normalization);
        }
        if request.with_timestamps {
            body["with_timestamps"] = serde_json::json!(true);
        }

        let response = self
            .client
            .post(format!("{}/v1/tts", self.settings.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(http::network_err)?;
        if !response.status().is_success() {
            return Err(http::error_for("xai", response).await);
        }
        let mime = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_else(|| mime_for_codec(request.codec))
            .to_string();
        let bytes = response.bytes().await.map_err(http::network_err)?;
        if request.with_timestamps || mime.contains("json") {
            let mut value: serde_json::Value =
                serde_json::from_slice(&bytes).map_err(|e| CoreError::Provider {
                    provider: "xai".into(),
                    message: format!("decoding timestamp response: {e}"),
                })?;
            let encoded = audio_base64(&value).ok_or_else(|| CoreError::Provider {
                provider: "xai".into(),
                message: "timestamp response did not include audio".into(),
            })?;
            let audio = base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .map_err(|e| CoreError::Provider {
                    provider: "xai".into(),
                    message: format!("decoding timestamp audio: {e}"),
                })?;
            redact_audio_fields(&mut value);
            return Ok(AudioOutput {
                bytes: audio,
                mime: mime_for_codec(request.codec).to_string(),
                codec: request.codec,
                provider: ProviderId::Xai,
                duration: None,
                metadata: Some(value),
            });
        }
        Ok(AudioOutput {
            bytes: bytes.to_vec(),
            mime,
            codec: request.codec,
            provider: ProviderId::Xai,
            duration: None,
            metadata: None,
        })
    }
}

#[derive(Debug, Deserialize)]
struct WsEvent {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    words: Option<Vec<SttWord>>,
    #[serde(default)]
    is_final: Option<bool>,
    #[serde(default)]
    speech_final: Option<bool>,
    #[serde(default)]
    duration: Option<f64>,
    #[serde(default)]
    audio: Option<String>,
    #[serde(default)]
    message: Option<String>,
}

#[async_trait]
impl StreamingTranscriber for XaiProvider {
    async fn stream_transcribe(
        &self,
        request: StreamTranscribeRequest,
    ) -> CoreResult<TranscriptStream> {
        let mut url = format!(
            "{}/v1/stt?sample_rate={}&encoding=pcm",
            self.settings.ws_url, request.sample_rate
        );
        if request.interim_results {
            url.push_str("&interim_results=true");
        }
        if request.smart_turn {
            url.push_str("&smart_turn=true");
        }
        if let Some(timeout) = request.smart_turn_timeout_ms {
            url.push_str(&format!("&smart_turn_timeout_ms={timeout}"));
        }
        if let Some(language) = &request.language {
            url.push_str(&format!(
                "&language={}",
                urlencoding::encode(&language.primary())
            ));
        }
        if request.diarize {
            url.push_str("&diarize=true");
        }
        for term in &request.keyterms {
            url.push_str(&format!("&keyterm={}", urlencoding::encode(term)));
        }

        let ws_request = self.ws_request(&url)?;
        let (socket, _) = connect_async(ws_request)
            .await
            .map_err(|e| CoreError::Network(format!("websocket connect: {e}")))?;
        let (mut sink, mut stream) = socket.split();

        // The server signals readiness before it will accept audio.
        loop {
            match stream.next().await {
                Some(Ok(Message::Text(text))) => {
                    let event: WsEvent =
                        serde_json::from_str(&text).map_err(|e| CoreError::Provider {
                            provider: "xai".into(),
                            message: format!("decoding ws event: {e}"),
                        })?;
                    match event.kind.as_str() {
                        "transcript.created" => break,
                        "error" => {
                            return Err(CoreError::Provider {
                                provider: "xai".into(),
                                message: event.message.unwrap_or_else(|| "ws error".into()),
                            })
                        }
                        _ => continue,
                    }
                }
                Some(Ok(_)) => continue,
                Some(Err(e)) => return Err(CoreError::Network(format!("websocket: {e}"))),
                None => {
                    return Err(CoreError::Network(
                        "websocket closed before transcript.created".into(),
                    ))
                }
            }
        }

        // Pump audio into the socket.
        let mut audio = request.audio;
        tokio::spawn(async move {
            while let Some(chunk) = audio.next().await {
                if sink.send(Message::Binary(chunk.0)).await.is_err() {
                    return;
                }
            }
            let _ = sink
                .send(Message::Text(r#"{"type":"audio.done"}"#.to_string()))
                .await;
        });

        // Surface transcript events.
        let (tx, rx) = futures::channel::mpsc::unbounded();
        tokio::spawn(async move {
            while let Some(message) = stream.next().await {
                let text = match message {
                    Ok(Message::Text(text)) => text,
                    Ok(Message::Close(_)) | Err(_) => break,
                    Ok(_) => continue,
                };
                let event: WsEvent = match serde_json::from_str(&text) {
                    Ok(event) => event,
                    Err(e) => {
                        let _ = tx.unbounded_send(Err(CoreError::Provider {
                            provider: "xai".into(),
                            message: format!("decoding ws event: {e}"),
                        }));
                        continue;
                    }
                };
                match event.kind.as_str() {
                    "transcript.partial" => {
                        let _ = tx.unbounded_send(Ok(TranscriptEvent::Partial {
                            text: event.text.unwrap_or_default(),
                            words: map_words(event.words.unwrap_or_default()),
                            is_final: event.is_final.unwrap_or(false),
                            speech_final: event.speech_final.unwrap_or(false),
                        }));
                    }
                    "transcript.done" => {
                        let mut transcript =
                            Transcript::new(ProviderId::Xai, event.text.unwrap_or_default());
                        transcript.model = Some("grok-stt".to_string());
                        transcript.duration = event.duration;
                        transcript.words = map_words(event.words.unwrap_or_default());
                        transcript.segments =
                            ov_core::domain::segments_from_words(&transcript.words);
                        let _ = tx.unbounded_send(Ok(TranscriptEvent::Done(transcript)));
                        break;
                    }
                    "error" => {
                        let _ = tx.unbounded_send(Err(CoreError::Provider {
                            provider: "xai".into(),
                            message: event.message.unwrap_or_else(|| "ws error".into()),
                        }));
                    }
                    _ => {}
                }
            }
        });

        Ok(Box::pin(rx))
    }
}

#[async_trait]
impl StreamingSpeechSynthesizer for XaiProvider {
    async fn stream_synthesize(&self, request: SpeechRequest) -> CoreResult<AudioEventStream> {
        let voice = request
            .voice
            .clone()
            .unwrap_or_else(|| self.settings.tts_voice.clone());
        let language = request
            .language
            .as_ref()
            .map(Language::code)
            .unwrap_or("auto")
            .to_string();
        let (codec, sample_rate) = codec_fields(request.codec, request.sample_rate)?;
        let url = format!(
            "{}/v1/tts?voice={}&language={}&codec={}&sample_rate={}",
            self.settings.ws_url,
            urlencoding::encode(&voice),
            urlencoding::encode(&language),
            codec,
            sample_rate,
        );
        let mut url = url;
        if let Some(bit_rate) = request.bit_rate {
            url.push_str(&format!("&bit_rate={bit_rate}"));
        }
        if let Some(latency) = request.optimize_streaming_latency {
            url.push_str(&format!("&optimize_streaming_latency={latency}"));
        }

        let ws_request = self.ws_request(&url)?;
        let (socket, _) = connect_async(ws_request)
            .await
            .map_err(|e| CoreError::Network(format!("websocket connect: {e}")))?;
        let (mut sink, mut stream) = socket.split();

        // Feed the text (a single utterance in v0.1) and close the turn.
        let delta = serde_json::json!({ "type": "text.delta", "delta": request.text });
        sink.send(Message::Text(delta.to_string()))
            .await
            .map_err(|e| CoreError::Network(format!("websocket send: {e}")))?;
        sink.send(Message::Text(r#"{"type":"text.done"}"#.to_string()))
            .await
            .map_err(|e| CoreError::Network(format!("websocket send: {e}")))?;

        let (tx, rx) = futures::channel::mpsc::unbounded();
        tokio::spawn(async move {
            while let Some(message) = stream.next().await {
                let text = match message {
                    Ok(Message::Text(text)) => text,
                    Ok(Message::Close(_)) | Err(_) => break,
                    Ok(_) => continue,
                };
                let event: WsEvent = match serde_json::from_str(&text) {
                    Ok(event) => event,
                    Err(e) => {
                        let _ = tx.unbounded_send(Err(CoreError::Provider {
                            provider: "xai".into(),
                            message: format!("decoding ws event: {e}"),
                        }));
                        continue;
                    }
                };
                match event.kind.as_str() {
                    "audio.delta" => {
                        let encoded = event.audio.unwrap_or_default();
                        match base64::engine::general_purpose::STANDARD.decode(encoded) {
                            Ok(bytes) => {
                                let _ = tx.unbounded_send(Ok(AudioEvent::Chunk(bytes)));
                            }
                            Err(e) => {
                                let _ = tx.unbounded_send(Err(CoreError::Provider {
                                    provider: "xai".into(),
                                    message: format!("decoding audio chunk: {e}"),
                                }));
                            }
                        }
                    }
                    "audio.done" => {
                        let _ = tx.unbounded_send(Ok(AudioEvent::Done));
                        break;
                    }
                    "error" => {
                        let _ = tx.unbounded_send(Err(CoreError::Provider {
                            provider: "xai".into(),
                            message: event.message.unwrap_or_else(|| "ws error".into()),
                        }));
                    }
                    _ => {}
                }
            }
        });

        Ok(Box::pin(rx))
    }
}
