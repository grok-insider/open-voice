//! Hermetic OpenAI adapter tests against a mock HTTP server.

use ov_core::domain::{AudioSource, SpeechRequest, TranscribeRequest};
use ov_core::ports::{BatchSpeechSynthesizer, BatchTranscriber, Provider};
use ov_core::{CoreError, ProviderId};
use ov_providers::{OpenAiProvider, OpenAiSettings};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn provider(base_url: String) -> OpenAiProvider {
    OpenAiProvider::new(
        "test-key",
        OpenAiSettings {
            base_url,
            ..Default::default()
        },
    )
}

fn source() -> AudioSource {
    AudioSource::Bytes {
        data: b"fake-audio".to_vec(),
        file_name: "clip.wav".into(),
        mime: "audio/wav".into(),
    }
}

#[tokio::test]
async fn transcribe_maps_verbose_json() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "task": "transcribe",
            "language": "spanish",
            "duration": 4.2,
            "text": "Hola mundo. Segunda frase",
            "segments": [
                {"id": 0, "seek": 0, "start": 0.0, "end": 0.9, "text": " Hola mundo."},
                {"id": 1, "seek": 0, "start": 3.0, "end": 3.9, "text": " Segunda frase"}
            ],
            "words": [
                {"word": "Hola", "start": 0.0, "end": 0.4},
                {"word": "mundo.", "start": 0.5, "end": 0.9}
            ]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let mut request = TranscribeRequest::new(source());
    request.language = Some(ov_core::domain::Language::new("es"));
    let transcript = provider(server.uri()).transcribe(request).await.unwrap();

    assert_eq!(transcript.provider, ProviderId::Openai);
    assert_eq!(transcript.text, "Hola mundo. Segunda frase");
    assert_eq!(transcript.language.as_deref(), Some("spanish"));
    assert_eq!(transcript.duration, Some(4.2));
    assert_eq!(transcript.segments.len(), 2);
    assert_eq!(transcript.segments[0].text, "Hola mundo.");
    assert_eq!(transcript.words.len(), 2);

    // The multipart body must carry model + verbose_json for whisper-1.
    let received: Vec<Request> = server.received_requests().await.unwrap();
    let body = String::from_utf8_lossy(&received[0].body);
    assert!(body.contains("whisper-1"));
    assert!(body.contains("verbose_json"));
    assert!(body.contains("timestamp_granularities"));
    assert!(body.contains("name=\"language\""));
}

#[tokio::test]
async fn transcribe_maps_auth_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("{\"error\":\"bad key\"}"))
        .mount(&server)
        .await;

    let err = provider(server.uri())
        .transcribe(TranscribeRequest::new(source()))
        .await
        .unwrap_err();
    assert!(matches!(err, CoreError::Auth { .. }), "got {err:?}");
}

#[tokio::test]
async fn transcribe_maps_rate_limit() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/transcriptions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("slow down"))
        .mount(&server)
        .await;

    let err = provider(server.uri())
        .transcribe(TranscribeRequest::new(source()))
        .await
        .unwrap_err();
    assert!(matches!(err, CoreError::RateLimited { .. }), "got {err:?}");
}

#[tokio::test]
async fn synthesize_posts_json_and_returns_bytes() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/audio/speech"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"MP3DATA".to_vec()))
        .expect(1)
        .mount(&server)
        .await;

    let mut request = SpeechRequest::new("Hola mundo");
    request.instructions = Some("cheerful".into());
    let output = provider(server.uri()).synthesize(request).await.unwrap();

    assert_eq!(output.bytes, b"MP3DATA");
    assert_eq!(output.mime, "audio/mpeg");

    let received: Vec<Request> = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(body["model"], "gpt-4o-mini-tts");
    assert_eq!(body["voice"], "marin");
    assert_eq!(body["input"], "Hola mundo");
    assert_eq!(body["response_format"], "mp3");
    assert_eq!(body["instructions"], "cheerful");
}

#[test]
fn capabilities_reflect_whisper_limits() {
    let p = provider("http://unused".into());
    let caps = p.capabilities();
    assert!(caps.batch_stt && caps.batch_tts);
    assert!(caps.word_timestamps);
    assert_eq!(caps.max_upload_bytes, Some(25 * 1024 * 1024));
    assert!(!caps.accepts_extension("ogg"));
    assert!(caps.accepts_extension("wav"));
}
