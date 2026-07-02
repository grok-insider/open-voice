//! Declarative provider capabilities. The engine validates a request against
//! these *before* any network call, so users get actionable errors ("openai
//! does not accept .ogg — transcoding first") instead of opaque 4xx bodies.

use serde::{Deserialize, Serialize};

use crate::domain::TranscribeRequest;
use crate::error::{CoreError, CoreResult};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    pub batch_stt: bool,
    pub batch_tts: bool,
    pub streaming_stt: bool,
    pub streaming_tts: bool,
    pub word_timestamps: bool,
    pub segment_timestamps: bool,
    pub diarization: bool,
    pub multichannel: bool,
    pub keyterms: bool,
    pub prompt: bool,
    /// Max upload size for batch STT, if the provider documents one.
    pub max_upload_bytes: Option<u64>,
    /// Accepted input file extensions for batch STT (lowercase, no dot).
    /// Empty means "anything the provider can decode".
    pub stt_input_extensions: Vec<String>,
}

impl ProviderCapabilities {
    pub fn accepts_extension(&self, ext: &str) -> bool {
        self.stt_input_extensions.is_empty()
            || self
                .stt_input_extensions
                .iter()
                .any(|e| e.eq_ignore_ascii_case(ext))
    }

    /// Validate a batch STT request against this capability set.
    /// `input_len` is the source size in bytes when known.
    pub fn validate_transcribe(
        &self,
        provider: &str,
        request: &TranscribeRequest,
        input_len: Option<u64>,
    ) -> CoreResult<()> {
        if !self.batch_stt {
            return Err(CoreError::Unsupported {
                provider: provider.to_string(),
                message: "batch speech-to-text is not supported".into(),
            });
        }
        if request.diarize && !self.diarization {
            return Err(CoreError::Unsupported {
                provider: provider.to_string(),
                message: "speaker diarization is not supported".into(),
            });
        }
        if !request.keyterms.is_empty() && !self.keyterms {
            return Err(CoreError::Unsupported {
                provider: provider.to_string(),
                message: "keyterm biasing is not supported".into(),
            });
        }
        if request.prompt.is_some() && !self.prompt {
            return Err(CoreError::Unsupported {
                provider: provider.to_string(),
                message: "prompting is not supported".into(),
            });
        }
        if let (Some(max), Some(len)) = (self.max_upload_bytes, input_len) {
            if len > max {
                return Err(CoreError::Unsupported {
                    provider: provider.to_string(),
                    message: format!("input is {len} bytes but the provider limit is {max} bytes"),
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{AudioSource, TranscribeRequest};

    fn caps() -> ProviderCapabilities {
        ProviderCapabilities {
            batch_stt: true,
            diarization: false,
            keyterms: false,
            prompt: false,
            max_upload_bytes: Some(100),
            stt_input_extensions: vec!["wav".into(), "mp3".into()],
            ..Default::default()
        }
    }

    fn req() -> TranscribeRequest {
        TranscribeRequest::new(AudioSource::Bytes {
            data: vec![0; 10],
            file_name: "a.wav".into(),
            mime: "audio/wav".into(),
        })
    }

    #[test]
    fn rejects_diarization_when_unsupported() {
        let mut r = req();
        r.diarize = true;
        let err = caps().validate_transcribe("p", &r, None).unwrap_err();
        assert!(matches!(err, CoreError::Unsupported { .. }));
    }

    #[test]
    fn rejects_oversized_input() {
        let err = caps()
            .validate_transcribe("p", &req(), Some(101))
            .unwrap_err();
        assert!(matches!(err, CoreError::Unsupported { .. }));
        caps().validate_transcribe("p", &req(), Some(100)).unwrap();
    }

    #[test]
    fn extension_check() {
        assert!(caps().accepts_extension("WAV"));
        assert!(!caps().accepts_extension("ogg"));
        let open = ProviderCapabilities::default();
        assert!(open.accepts_extension("anything"));
    }
}
