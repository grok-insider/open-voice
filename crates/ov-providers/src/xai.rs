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
        other => {
            return Err(CoreError::Unsupported {
                provider: "xai".into(),
                message: format!("output codec '{}' (use mp3, wav, or pcm)", other.as_str()),
            })
        }
    };
    Ok((name.to_string(), sample_rate.unwrap_or(24_000)))
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
        let mut body = serde_json::json!({
            "text": request.text,
            "language": language,
            "voice_id": voice,
            "output_format": { "codec": codec, "sample_rate": sample_rate },
        });
        if let Some(speed) = request.speed {
            body["speed"] = serde_json::json!(speed);
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
            .unwrap_or("application/octet-stream")
            .to_string();
        let bytes = response.bytes().await.map_err(http::network_err)?;
        Ok(AudioOutput {
            bytes: bytes.to_vec(),
            mime,
            codec: request.codec,
            provider: ProviderId::Xai,
            duration: None,
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
