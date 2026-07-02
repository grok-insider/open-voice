//! Hermetic ElevenLabs adapter tests against a mock HTTP server.

use ov_core::domain::{AudioCodec, AudioSource, SpeechRequest, TranscribeRequest};
use ov_core::ports::{BatchSpeechSynthesizer, BatchTranscriber, Provider};
use ov_core::{CoreError, ProviderId};
use ov_providers::{ElevenLabsProvider, ElevenLabsSettings};
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn provider(base_url: String) -> ElevenLabsProvider {
    ElevenLabsProvider::new(
        "xi-test-key",
        ElevenLabsSettings {
            base_url,
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
async fn transcribe_maps_words_and_speakers() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/speech-to-text"))
        .and(header("xi-api-key", "xi-test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "language_code": "es",
            "language_probability": 0.99,
            "text": "Hola mundo.",
            "words": [
                {"text": "Hola", "type": "word", "start": 0.0, "end": 0.4, "speaker_id": "speaker_0"},
                {"text": " ", "type": "spacing", "start": 0.4, "end": 0.5},
                {"text": "mundo.", "type": "word", "start": 0.5, "end": 0.9, "speaker_id": "speaker_0"},
                {"text": "(laughter)", "type": "audio_event", "start": 1.0, "end": 1.5}
            ]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let mut request = TranscribeRequest::new(source());
    request.diarize = true;
    request.keyterms = vec!["mundo".into()];
    let transcript = provider(server.uri()).transcribe(request).await.unwrap();

    assert_eq!(transcript.provider, ProviderId::Elevenlabs);
    assert_eq!(transcript.text, "Hola mundo.");
    assert_eq!(transcript.language.as_deref(), Some("es"));
    // spacing + audio_event entries are dropped.
    assert_eq!(transcript.words.len(), 2);
    assert_eq!(transcript.words[0].speaker.as_deref(), Some("speaker_0"));
    assert_eq!(transcript.segments.len(), 1);
    assert_eq!(transcript.segments[0].text, "Hola mundo.");
    assert_eq!(transcript.duration, Some(0.9));

    let received: Vec<Request> = server.received_requests().await.unwrap();
    let body = String::from_utf8_lossy(&received[0].body);
    assert!(body.contains("scribe_v2"));
    assert!(body.contains("name=\"diarize\""));
    assert!(body.contains("name=\"keyterms\""));
}

#[tokio::test]
async fn synthesize_uses_voice_path_and_output_format() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"))
        .and(query_param("output_format", "mp3_44100_128"))
        .and(header("xi-api-key", "xi-test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"MP3DATA".to_vec()))
        .expect(1)
        .mount(&server)
        .await;

    let output = provider(server.uri())
        .synthesize(SpeechRequest::new("Hola"))
        .await
        .unwrap();
    assert_eq!(output.bytes, b"MP3DATA");

    let received: Vec<Request> = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
    assert_eq!(body["model_id"], "eleven_multilingual_v2");
    assert_eq!(body["text"], "Hola");
    // multilingual_v2 must NOT receive language_code.
    assert!(body.get("language_code").is_none());
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
fn capabilities_advertise_diarization() {
    let caps = provider("http://unused".into()).capabilities();
    assert!(caps.diarization && caps.keyterms && caps.word_timestamps);
    assert!(caps.accepts_extension("ogg"));
}
