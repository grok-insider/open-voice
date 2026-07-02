//! Hermetic xAI batch STT/TTS tests against a mock HTTP server.

use ov_core::domain::{AudioCodec, AudioSource, SpeechRequest, TranscribeRequest};
use ov_core::ports::{BatchSpeechSynthesizer, BatchTranscriber, Provider};
use ov_core::{CoreError, ProviderId};
use ov_providers::{XaiProvider, XaiSettings};
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
    let output = provider(server.uri()).synthesize(request).await.unwrap();
    assert_eq!(output.bytes, b"MP3DATA");
    assert_eq!(output.mime, "audio/mpeg");

    let received: Vec<Request> = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(body["text"], "Hola mundo");
    assert_eq!(body["language"], "es");
    assert_eq!(body["voice_id"], "eve");
    assert_eq!(body["output_format"]["codec"], "mp3");
    let speed = body["speed"].as_f64().unwrap();
    assert!((speed - 1.1).abs() < 1e-6, "speed {speed}");
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
