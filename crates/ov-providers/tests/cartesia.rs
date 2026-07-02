//! Hermetic Cartesia adapter tests against a mock HTTP server.

use ov_core::domain::{AudioSource, SpeechRequest, TranscribeRequest};
use ov_core::ports::{BatchSpeechSynthesizer, BatchTranscriber, Provider};
use ov_core::{CoreError, ProviderId};
use ov_providers::{CartesiaProvider, CartesiaSettings};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn provider(base_url: String, voice: &str) -> CartesiaProvider {
    CartesiaProvider::new(
        "cartesia-key",
        CartesiaSettings {
            base_url,
            tts_voice: voice.to_string(),
            ..Default::default()
        },
    )
}

fn source() -> AudioSource {
    AudioSource::Bytes {
        data: b"fake-audio".to_vec(),
        file_name: "clip.ogg".into(),
        mime: "audio/ogg".into(),
    }
}

#[tokio::test]
async fn transcribe_maps_words_and_duration() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/stt"))
        .and(header("authorization", "Bearer cartesia-key"))
        .and(header("Cartesia-Version", "2026-03-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "type": "transcript",
            "text": "Hola mundo.",
            "language": "es",
            "duration": 1.2,
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
    let transcript = provider(server.uri(), "")
        .transcribe(request)
        .await
        .unwrap();

    assert_eq!(transcript.provider, ProviderId::Cartesia);
    assert_eq!(transcript.model.as_deref(), Some("ink-whisper"));
    assert_eq!(transcript.duration, Some(1.2));
    assert_eq!(transcript.words.len(), 2);
    assert_eq!(transcript.segments.len(), 1);

    let received: Vec<Request> = server.received_requests().await.unwrap();
    let body = String::from_utf8_lossy(&received[0].body);
    assert!(body.contains("ink-whisper"));
    assert!(body.contains("timestamp_granularities"));
}

#[tokio::test]
async fn synthesize_builds_sonic_payload() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/tts/bytes"))
        .and(header("Cartesia-Version", "2026-03-01"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"MP3DATA".to_vec()))
        .expect(1)
        .mount(&server)
        .await;

    let mut request = SpeechRequest::new("Hola mundo");
    request.language = Some(ov_core::domain::Language::new("es"));
    let output = provider(server.uri(), "voice-123")
        .synthesize(request)
        .await
        .unwrap();
    assert_eq!(output.bytes, b"MP3DATA");

    let received: Vec<Request> = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(body["model_id"], "sonic-3.5");
    assert_eq!(body["voice"]["mode"], "id");
    assert_eq!(body["voice"]["id"], "voice-123");
    assert_eq!(body["output_format"]["container"], "mp3");
    assert_eq!(body["language"], "es");
}

#[tokio::test]
async fn synthesize_without_voice_fails_fast() {
    let err = provider("http://unused".into(), "")
        .synthesize(SpeechRequest::new("Hola"))
        .await
        .unwrap_err();
    assert!(
        matches!(err, CoreError::NotConfigured { .. }),
        "got {err:?}"
    );
}

#[test]
fn capabilities_accept_ogg() {
    let caps = provider("http://unused".into(), "").capabilities();
    assert!(caps.batch_stt && caps.batch_tts);
    assert!(caps.accepts_extension("ogg"));
    assert!(!caps.diarization);
}
