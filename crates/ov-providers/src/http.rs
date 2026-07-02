//! Shared HTTP plumbing: one client shape and one status→CoreError mapping so
//! every adapter reports failures the same way.

use std::time::Duration;

use ov_core::domain::AudioSource;
use ov_core::{CoreError, CoreResult};

pub(crate) fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent(concat!("open-voice/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(15))
        // Big uploads / long syntheses: generous overall timeout.
        .timeout(Duration::from_secs(15 * 60))
        .build()
        .expect("reqwest client")
}

pub(crate) fn network_err(e: reqwest::Error) -> CoreError {
    CoreError::Network(e.to_string())
}

/// Map a non-success response to a `CoreError`, consuming the body for
/// context (truncated so errors stay readable).
pub(crate) async fn error_for(provider: &str, response: reqwest::Response) -> CoreError {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    let mut message = body.trim().to_string();
    if message.len() > 600 {
        message.truncate(600);
        message.push('…');
    }
    if message.is_empty() {
        message = status.to_string();
    } else {
        message = format!("HTTP {}: {message}", status.as_u16());
    }
    match status.as_u16() {
        401 | 403 => CoreError::Auth {
            provider: provider.to_string(),
            message,
        },
        429 => CoreError::RateLimited {
            provider: provider.to_string(),
            message,
        },
        _ => CoreError::Provider {
            provider: provider.to_string(),
            message,
        },
    }
}

/// Materialize an `AudioSource` into (bytes, file name, mime) for a multipart
/// upload.
pub(crate) async fn source_parts(source: &AudioSource) -> CoreResult<(Vec<u8>, String, String)> {
    match source {
        AudioSource::File(path) => {
            let data = tokio::fs::read(path)
                .await
                .map_err(|e| CoreError::Io(format!("reading {}: {e}", path.display())))?;
            let name = source.file_name();
            let ext = source.extension().unwrap_or_default();
            Ok((data, name, guess_mime(&ext).to_string()))
        }
        AudioSource::Bytes {
            data,
            file_name,
            mime,
        } => Ok((data.clone(), file_name.clone(), mime.clone())),
    }
}

fn guess_mime(ext: &str) -> &'static str {
    match ext {
        "mp3" | "mpga" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" | "oga" => "audio/ogg",
        "opus" => "audio/opus",
        "flac" => "audio/flac",
        "m4a" => "audio/mp4",
        "mp4" => "video/mp4",
        "aac" => "audio/aac",
        "webm" => "audio/webm",
        "mkv" => "video/x-matroska",
        _ => "application/octet-stream",
    }
}
