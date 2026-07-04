//! Hermetic xAI batch STT/TTS tests against a mock HTTP server.

use base64::Engine as _;
use ov_core::domain::{AudioCodec, AudioSource, SpeechRequest, TranscribeRequest};
use ov_core::ports::{BatchSpeechSynthesizer, BatchTranscriber, Provider};
use ov_core::{CoreError, ProviderId};
use ov_providers::{CustomVoiceCreateRequest, CustomVoiceUpdateRequest, XaiProvider, XaiSettings};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn provider(base_url: String) -> XaiProvider {
    XaiProvider::new(
        "xai-key",
        XaiSettings {
            base_url,
            ..Default::default()
        },
    )
}

fn source() -> AudioSource {
    AudioSource::Bytes {
        data: b"fake-ogg-audio".to_vec(),
        file_name: "WhatsApp Ptt 2026-07-02 at 14.56.03.ogg".into(),
        mime: "audio/ogg".into(),
    }
}

#[tokio::test]
async fn transcribe_maps_words_and_diarization() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/stt"))
        .and(header("authorization", "Bearer xai-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "text": "Hola mundo. Segunda frase",
            "language": "Spanish",
            "duration": 4.2,
            "words": [
                {"text": "Hola", "start": 0.0, "end": 0.4, "speaker": 0},
                {"text": "mundo.", "start": 0.5, "end": 0.9, "speaker": 0},
                {"text": "Segunda", "start": 3.0, "end": 3.4, "speaker": 1},
                {"text": "frase", "start": 3.5, "end": 3.9, "speaker": 1}
            ]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let mut request = TranscribeRequest::new(source());
    request.language = Some(ov_core::domain::Language::new("es"));
    request.diarize = true;
    request.keyterms = vec!["Hola".into()];
    let transcript = provider(server.uri()).transcribe(request).await.unwrap();

    assert_eq!(transcript.provider, ProviderId::Xai);
    assert_eq!(transcript.duration, Some(4.2));
    assert_eq!(transcript.words.len(), 4);
    assert_eq!(transcript.words[0].speaker.as_deref(), Some("speaker_0"));
    // Speaker change forces a segment split.
    assert_eq!(transcript.segments.len(), 2);

    let received: Vec<Request> = server.received_requests().await.unwrap();
    let body = String::from_utf8_lossy(&received[0].body);
    assert!(body.contains("name=\"language\""));
    assert!(body.contains("name=\"format\""));
    assert!(body.contains("name=\"diarize\""));
    assert!(body.contains("name=\"keyterm\""));
    // xAI requires the file part to be the LAST multipart field.
    let file_pos = body.find("name=\"file\"").expect("file field present");
    for field in ["name=\"language\"", "name=\"diarize\"", "name=\"keyterm\""] {
        assert!(body.find(field).unwrap() < file_pos, "{field} after file");
    }
}

#[tokio::test]
async fn synthesize_builds_tts_payload() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/tts"))
        .and(header("authorization", "Bearer xai-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "audio/mpeg")
                .set_body_bytes(b"MP3DATA".to_vec()),
        )
        .expect(1)
        .mount(&server)
        .await;

    let mut request = SpeechRequest::new("Hola mundo");
    request.language = Some(ov_core::domain::Language::new("es"));
    request.speed = Some(1.1);
    request.model = Some("grok-voice-latest".into());
    request.bit_rate = Some(128_000);
    request.optimize_streaming_latency = Some(1);
    request.text_normalization = Some(true);
    let output = provider(server.uri()).synthesize(request).await.unwrap();
    assert_eq!(output.bytes, b"MP3DATA");
    assert_eq!(output.mime, "audio/mpeg");

    let received: Vec<Request> = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(body["text"], "Hola mundo");
    assert_eq!(body["language"], "es");
    assert_eq!(body["voice_id"], "eve");
    assert_eq!(body["model"], "grok-voice-latest");
    assert_eq!(body["output_format"]["codec"], "mp3");
    assert_eq!(body["output_format"]["bit_rate"], 128000);
    assert_eq!(body["optimize_streaming_latency"], 1);
    assert_eq!(body["text_normalization"], true);
    let speed = body["speed"].as_f64().unwrap();
    assert!((speed - 1.1).abs() < 1e-6, "speed {speed}");
}

#[tokio::test]
async fn synthesize_maps_timestamp_json_response() {
    let server = MockServer::start().await;
    let audio = base64::engine::general_purpose::STANDARD.encode(b"MP3DATA");
    Mock::given(method("POST"))
        .and(path("/v1/tts"))
        .and(header("authorization", "Bearer xai-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/json")
                .set_body_json(serde_json::json!({
                    "audio": audio,
                    "characters": [{"char": "H", "start": 0.0, "end": 0.1}]
                })),
        )
        .expect(1)
        .mount(&server)
        .await;

    let mut request = SpeechRequest::new("Hola");
    request.with_timestamps = true;
    let output = provider(server.uri()).synthesize(request).await.unwrap();
    assert_eq!(output.bytes, b"MP3DATA");
    let metadata = output.metadata.expect("timestamp metadata");
    assert_eq!(metadata["audio"], "<base64 audio omitted>");
    assert_eq!(metadata["characters"][0]["char"], "H");

    let received: Vec<Request> = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(body["with_timestamps"], true);
}

#[tokio::test]
async fn synthesize_rejects_unsupported_codec() {
    let mut request = SpeechRequest::new("Hola");
    request.codec = AudioCodec::Flac;
    let err = provider("http://unused".into())
        .synthesize(request)
        .await
        .unwrap_err();
    assert!(matches!(err, CoreError::Unsupported { .. }), "got {err:?}");
}

#[test]
fn capabilities_are_full_featured() {
    let caps = provider("http://unused".into()).capabilities();
    assert!(caps.batch_stt && caps.batch_tts && caps.streaming_stt && caps.streaming_tts);
    assert!(caps.diarization && caps.keyterms && caps.multichannel);
    assert!(caps.accepts_extension("ogg"));
    assert_eq!(caps.max_upload_bytes, Some(500 * 1024 * 1024));
}

#[tokio::test]
async fn custom_voices_list_and_get() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/custom-voices"))
        .and(header("authorization", "Bearer xai-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "voices": [{
                "voice_id": "abc123xy",
                "name": "Narrator",
                "language": "en",
                "tone": "warm"
            }],
            "pagination_token": "next"
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v1/custom-voices/abc123xy"))
        .and(header("authorization", "Bearer xai-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "voice_id": "abc123xy",
            "name": "Narrator",
            "description": "Warm narration"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = provider(server.uri());
    let list = provider.list_custom_voices(Some(50), None).await.unwrap();
    assert_eq!(list.voices[0].voice_id, "abc123xy");
    assert_eq!(list.pagination_token.as_deref(), Some("next"));

    let voice = provider.get_custom_voice("abc123xy").await.unwrap();
    assert_eq!(voice.name.as_deref(), Some("Narrator"));
}

#[tokio::test]
async fn custom_voice_create_update_delete() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/custom-voices"))
        .and(header("authorization", "Bearer xai-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "voice_id": "abc123xy",
            "name": "Narrator",
            "language": "en"
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("PATCH"))
        .and(path("/v1/custom-voices/abc123xy"))
        .and(header("authorization", "Bearer xai-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "voice_id": "abc123xy",
            "name": "Narrator 2",
            "tone": "calm"
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/v1/custom-voices/abc123xy"))
        .and(header("authorization", "Bearer xai-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "deleted": true
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = provider(server.uri());
    let created = provider
        .create_custom_voice(CustomVoiceCreateRequest {
            file: source(),
            name: "Narrator".into(),
            description: Some("Warm narration".into()),
            gender: Some("female".into()),
            accent: None,
            age: None,
            language: Some("en".into()),
            use_case: Some("narration".into()),
            tone: Some("warm".into()),
        })
        .await
        .unwrap();
    assert_eq!(created.voice_id, "abc123xy");

    let updated = provider
        .update_custom_voice(
            "abc123xy",
            CustomVoiceUpdateRequest {
                name: Some("Narrator 2".into()),
                tone: Some("calm".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(updated.name.as_deref(), Some("Narrator 2"));

    assert!(provider.delete_custom_voice("abc123xy").await.unwrap());

    let received: Vec<Request> = server.received_requests().await.unwrap();
    let create_body = String::from_utf8_lossy(&received[0].body);
    assert!(create_body.contains("name=\"name\""));
    assert!(create_body.contains("name=\"file\""));
    let update_body: serde_json::Value = serde_json::from_slice(&received[1].body).unwrap();
    assert_eq!(update_body["name"], "Narrator 2");
    assert_eq!(update_body["tone"], "calm");
}
