//! Error vocabulary shared by every port. Adapters map their concrete errors
//! (HTTP status codes, ONNX failures, ffmpeg exits) into these variants at the
//! boundary so the engine can react uniformly (retry, fall back, report).

use thiserror::Error;

pub type CoreResult<T> = Result<T, CoreError>;

#[derive(Debug, Error)]
pub enum CoreError {
    /// Missing/invalid credentials (HTTP 401/403 or absent API key).
    #[error("authentication failed for {provider}: {message}")]
    Auth { provider: String, message: String },

    /// Provider throttled us (HTTP 429). The engine may retry or fall back.
    #[error("{provider} rate limited: {message}")]
    RateLimited { provider: String, message: String },

    /// The request is valid but this provider/engine cannot serve it
    /// (capability mismatch: diarization, language, file size, format...).
    #[error("unsupported by {provider}: {message}")]
    Unsupported { provider: String, message: String },

    /// The caller's input is invalid regardless of provider.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// The provider is not configured (no API key / model files missing).
    #[error("{provider} is not configured: {message}")]
    NotConfigured { provider: String, message: String },

    /// Upstream returned an error we don't model more precisely.
    #[error("{provider} error: {message}")]
    Provider { provider: String, message: String },

    /// Transport-level failure (DNS, TLS, connection reset...).
    #[error("network error: {0}")]
    Network(String),

    /// Local audio decode/probe failure.
    #[error("audio error: {0}")]
    Audio(String),

    /// Local filesystem failure.
    #[error("io error: {0}")]
    Io(String),
}

impl CoreError {
    /// Errors after which trying the next provider in an `auto` chain makes
    /// sense (versus errors that would fail identically everywhere).
    pub fn is_fallback_worthy(&self) -> bool {
        matches!(
            self,
            CoreError::Auth { .. }
                | CoreError::RateLimited { .. }
                | CoreError::Unsupported { .. }
                | CoreError::NotConfigured { .. }
                | CoreError::Provider { .. }
                | CoreError::Network(_)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_input_is_not_fallback_worthy() {
        assert!(!CoreError::InvalidInput("x".into()).is_fallback_worthy());
        assert!(!CoreError::Io("x".into()).is_fallback_worthy());
    }

    #[test]
    fn provider_side_errors_are_fallback_worthy() {
        let e = CoreError::RateLimited {
            provider: "xai".into(),
            message: "slow down".into(),
        };
        assert!(e.is_fallback_worthy());
    }
}
