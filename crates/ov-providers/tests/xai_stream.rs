//! xAI streaming tests against an in-process mock WebSocket server that
//! speaks the documented protocol.

use futures_util::{SinkExt, StreamExt};
use ov_core::domain::{Language, SpeechRequest};
use ov_core::ports::{
    AudioEvent, PcmChunk, StreamTranscribeRequest, StreamingSpeechSynthesizer,
    StreamingTranscriber, TranscriptEvent,
};
use ov_providers::{XaiProvider, XaiSettings};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;

async fn spawn_stt_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();
        ws.send(Message::Text(
            r#"{"type":"transcript.created"}"#.to_string(),
        ))
        .await
        .unwrap();

        let mut audio_bytes = 0usize;
        while let Some(Ok(message)) = ws.next().await {
            match message {
                Message::Binary(data) => audio_bytes += data.len(),
                Message::Text(text) => {
                    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
                    if value["type"] == "audio.done" {
                        break;
                    }
                }
                Message::Close(_) => return,
                _ => {}
            }
        }
        assert!(audio_bytes > 0, "server received no audio");

        ws.send(Message::Text(
            serde_json::json!({
                "type": "transcript.partial",
                "text": "Hola",
                "words": [{"text": "Hola", "start": 0.0, "end": 0.4}],
                "is_final": true,
                "speech_final": false,
            })
            .to_string(),
        ))
        .await
        .unwrap();
        ws.send(Message::Text(
            serde_json::json!({
                "type": "transcript.done",
                "text": "Hola mundo.",
                "duration": 1.0,
                "words": [
                    {"text": "Hola", "start": 0.0, "end": 0.4},
                    {"text": "mundo.", "start": 0.5, "end": 0.9}
                ],
            })
            .to_string(),
        ))
        .await
        .unwrap();
    });
    format!("ws://{addr}")
}

async fn spawn_tts_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut ws = tokio_tungstenite::accept_async(stream).await.unwrap();

        let mut got_text = String::new();
        while let Some(Ok(Message::Text(text))) = ws.next().await {
            let value: serde_json::Value = serde_json::from_str(&text).unwrap();
            match value["type"].as_str() {
                Some("text.delta") => {
                    got_text.push_str(value["delta"].as_str().unwrap_or_default())
                }
                Some("text.done") => break,
                _ => {}
            }
        }
        assert_eq!(got_text, "Hola mundo");

        use base64::Engine as _;
        let chunk = base64::engine::general_purpose::STANDARD.encode(b"AUDIO1");
        ws.send(Message::Text(
            serde_json::json!({"type": "audio.delta", "audio": chunk}).to_string(),
        ))
        .await
        .unwrap();
        ws.send(Message::Text(r#"{"type":"audio.done"}"#.to_string()))
            .await
            .unwrap();
    });
    format!("ws://{addr}")
}

fn provider(ws_url: String) -> XaiProvider {
    XaiProvider::new(
        "xai-key",
        XaiSettings {
            ws_url,
            ..Default::default()
        },
    )
}

#[tokio::test]
async fn streaming_stt_round_trip() {
    let ws_url = spawn_stt_server().await;
    let audio: ov_core::ports::PcmStream =
        Box::pin(futures::stream::iter(vec![PcmChunk(vec![0u8; 3200])]));
    let request = StreamTranscribeRequest {
        audio,
        sample_rate: 16_000,
        language: Some(Language::new("es")),
        diarize: false,
        keyterms: vec!["Hola".into()],
        interim_results: true,
    };

    let mut stream = provider(ws_url).stream_transcribe(request).await.unwrap();

    let mut partials = 0;
    let mut done: Option<ov_core::domain::Transcript> = None;
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            TranscriptEvent::Partial { text, is_final, .. } => {
                assert_eq!(text, "Hola");
                assert!(is_final);
                partials += 1;
            }
            TranscriptEvent::Done(transcript) => {
                done = Some(transcript);
            }
        }
    }
    assert_eq!(partials, 1);
    let transcript = done.expect("final transcript");
    assert_eq!(transcript.text, "Hola mundo.");
    assert_eq!(transcript.words.len(), 2);
    assert_eq!(transcript.duration, Some(1.0));
}

#[tokio::test]
async fn streaming_tts_round_trip() {
    let ws_url = spawn_tts_server().await;
    let mut request = SpeechRequest::new("Hola mundo");
    request.language = Some(Language::new("es"));

    let mut stream = provider(ws_url).stream_synthesize(request).await.unwrap();

    let mut audio = Vec::new();
    let mut finished = false;
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            AudioEvent::Chunk(bytes) => audio.extend_from_slice(&bytes),
            AudioEvent::Done => finished = true,
        }
    }
    assert!(finished);
    assert_eq!(audio, b"AUDIO1");
}
