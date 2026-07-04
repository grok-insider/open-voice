//! # ov-engine
//!
//! Use-cases on top of the `ov-core` ports: provider selection, capability
//! validation, automatic transcoding, and `auto` fallback chains. Depends
//! only on `ov-core` — concrete adapters are registered by the composition
//! root (the CLI), in preference order.

use std::path::PathBuf;
use std::sync::Arc;

use ov_core::domain::AudioSource;
use ov_core::domain::{AudioOutput, SpeechRequest, TranscribeRequest, Transcript};
use ov_core::ports::{
    AudioDecoder, AudioSpec, BatchSpeechSynthesizer, BatchTranscriber, StreamingSpeechSynthesizer,
    StreamingTranscriber,
};
use ov_core::{CoreError, CoreResult, ProviderId};

/// The composition of all registered adapters, in preference order.
#[derive(Default)]
pub struct Engine {
    stt: Vec<Arc<dyn BatchTranscriber>>,
    tts: Vec<Arc<dyn BatchSpeechSynthesizer>>,
    streaming_stt: Vec<Arc<dyn StreamingTranscriber>>,
    streaming_tts: Vec<Arc<dyn StreamingSpeechSynthesizer>>,
    decoder: Option<Arc<dyn AudioDecoder>>,
}

impl Engine {
    pub fn builder() -> EngineBuilder {
        EngineBuilder::default()
    }

    pub fn stt_providers(&self) -> &[Arc<dyn BatchTranscriber>] {
        &self.stt
    }

    pub fn tts_providers(&self) -> &[Arc<dyn BatchSpeechSynthesizer>] {
        &self.tts
    }

    pub fn streaming_stt_provider(
        &self,
        want: Option<ProviderId>,
    ) -> CoreResult<Arc<dyn StreamingTranscriber>> {
        pick_one(&self.streaming_stt, want, "streaming speech-to-text")
    }

    pub fn streaming_tts_provider(
        &self,
        want: Option<ProviderId>,
    ) -> CoreResult<Arc<dyn StreamingSpeechSynthesizer>> {
        pick_one(&self.streaming_tts, want, "streaming text-to-speech")
    }

    /// Batch STT with validation, automatic transcoding, and fallback.
    ///
    /// * `want = Some(id)` — use exactly that provider (still transcoding
    ///   input the provider can't ingest).
    /// * `want = None` — try every registered provider in registration order,
    ///   moving on after provider-side failures.
    pub async fn transcribe(
        &self,
        request: TranscribeRequest,
        want: Option<ProviderId>,
    ) -> CoreResult<Transcript> {
        let candidates: Vec<&Arc<dyn BatchTranscriber>> = match want {
            Some(id) => vec![self
                .stt
                .iter()
                .find(|p| p.id() == id)
                .ok_or_else(|| not_registered(id, "speech-to-text"))?],
            None => self.stt.iter().collect(),
        };
        if candidates.is_empty() {
            return Err(CoreError::NotConfigured {
                provider: "auto".into(),
                message: "no speech-to-text providers are configured (set an API key or install a local model)".into(),
            });
        }

        let auto = want.is_none();
        let mut last_err: Option<CoreError> = None;
        // Temp files created by transcoding, cleaned up when we're done.
        let mut temp_files: Vec<PathBuf> = Vec::new();

        let mut result = None;
        for provider in candidates {
            let caps = provider.capabilities();
            let mut attempt = request.clone();

            // Transcode when the provider can't ingest the container.
            let ext = attempt.source.extension().unwrap_or_default();
            if !ext.is_empty() && !caps.accepts_extension(&ext) {
                match self.transcode(&attempt.source).await {
                    Ok(path) => {
                        tracing::debug!(provider = %provider.id(), from = %ext, "transcoded input to wav");
                        temp_files.push(path.clone());
                        attempt.source = AudioSource::File(path);
                    }
                    Err(e) => {
                        if auto {
                            last_err = Some(e);
                            continue;
                        }
                        cleanup(&temp_files);
                        return Err(e);
                    }
                }
            }

            let input_len = source_len(&attempt.source);
            if let Err(e) = caps.validate_transcribe(provider.id().as_str(), &attempt, input_len) {
                if auto {
                    last_err = Some(e);
                    continue;
                }
                cleanup(&temp_files);
                return Err(e);
            }

            match provider.transcribe(attempt).await {
                Ok(transcript) => {
                    result = Some(transcript);
                    break;
                }
                Err(e) if auto && e.is_fallback_worthy() => {
                    tracing::warn!(provider = %provider.id(), error = %e, "provider failed, trying next");
                    last_err = Some(e);
                }
                Err(e) => {
                    cleanup(&temp_files);
                    return Err(e);
                }
            }
        }

        cleanup(&temp_files);
        result.ok_or_else(|| {
            last_err.unwrap_or_else(|| CoreError::NotConfigured {
                provider: "auto".into(),
                message: "no provider could handle the request".into(),
            })
        })
    }

    /// Batch TTS with `auto` fallback.
    pub async fn speak(
        &self,
        request: SpeechRequest,
        want: Option<ProviderId>,
    ) -> CoreResult<AudioOutput> {
        let candidates: Vec<&Arc<dyn BatchSpeechSynthesizer>> = match want {
            Some(id) => vec![self
                .tts
                .iter()
                .find(|p| p.id() == id)
                .ok_or_else(|| not_registered(id, "text-to-speech"))?],
            None => self.tts.iter().collect(),
        };
        if candidates.is_empty() {
            return Err(CoreError::NotConfigured {
                provider: "auto".into(),
                message: "no text-to-speech providers are configured (set an API key or install a local TTS model)".into(),
            });
        }

        let auto = want.is_none();
        let mut last_err: Option<CoreError> = None;
        for provider in candidates {
            match provider.synthesize(request.clone()).await {
                Ok(output) => return Ok(output),
                Err(e) if auto && e.is_fallback_worthy() => {
                    tracing::warn!(provider = %provider.id(), error = %e, "provider failed, trying next");
                    last_err = Some(e);
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.unwrap_or_else(|| CoreError::NotConfigured {
            provider: "auto".into(),
            message: "no provider could handle the request".into(),
        }))
    }

    async fn transcode(&self, source: &AudioSource) -> CoreResult<PathBuf> {
        let decoder = self.decoder.as_ref().ok_or_else(|| CoreError::Audio(
            "input format requires transcoding but no audio decoder is available (is ffmpeg installed?)".into(),
        ))?;
        let path = match source {
            AudioSource::File(path) => path.clone(),
            AudioSource::Bytes { .. } => {
                return Err(CoreError::Audio(
                    "transcoding in-memory sources is not supported".into(),
                ))
            }
        };
        decoder.decode_to_wav(&path, AudioSpec::STT_16K_MONO).await
    }
}

fn source_len(source: &AudioSource) -> Option<u64> {
    match source {
        AudioSource::File(path) => std::fs::metadata(path).map(|m| m.len()).ok(),
        AudioSource::Bytes { data, .. } => Some(data.len() as u64),
    }
}

fn cleanup(paths: &[PathBuf]) {
    for path in paths {
        let _ = std::fs::remove_file(path);
    }
}

fn not_registered(id: ProviderId, what: &str) -> CoreError {
    CoreError::NotConfigured {
        provider: id.to_string(),
        message: format!("provider is not configured for {what} (missing API key or model?)"),
    }
}

fn pick_one<T: ?Sized + ov_core::ports::Provider>(
    providers: &[Arc<T>],
    want: Option<ProviderId>,
    what: &str,
) -> CoreResult<Arc<T>> {
    match want {
        Some(id) => providers
            .iter()
            .find(|p| p.id() == id)
            .cloned()
            .ok_or_else(|| not_registered(id, what)),
        None => providers
            .first()
            .cloned()
            .ok_or_else(|| CoreError::NotConfigured {
                provider: "auto".into(),
                message: format!("no providers are configured for {what}"),
            }),
    }
}

/// Builder: the composition root registers adapters in preference order.
#[derive(Default)]
pub struct EngineBuilder {
    engine: Engine,
}

impl EngineBuilder {
    pub fn stt(mut self, provider: Arc<dyn BatchTranscriber>) -> Self {
        self.engine.stt.push(provider);
        self
    }

    pub fn tts(mut self, provider: Arc<dyn BatchSpeechSynthesizer>) -> Self {
        self.engine.tts.push(provider);
        self
    }

    pub fn streaming_stt(mut self, provider: Arc<dyn StreamingTranscriber>) -> Self {
        self.engine.streaming_stt.push(provider);
        self
    }

    pub fn streaming_tts(mut self, provider: Arc<dyn StreamingSpeechSynthesizer>) -> Self {
        self.engine.streaming_tts.push(provider);
        self
    }

    pub fn decoder(mut self, decoder: Arc<dyn AudioDecoder>) -> Self {
        self.engine.decoder = Some(decoder);
        self
    }

    pub fn build(self) -> Engine {
        self.engine
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use ov_core::capabilities::ProviderCapabilities;
    use ov_core::ports::Provider;
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    struct FakeStt {
        id: ProviderId,
        caps: ProviderCapabilities,
        result: Mutex<Option<CoreResult<Transcript>>>,
        calls: AtomicUsize,
        seen_ext: Mutex<Option<String>>,
    }

    impl FakeStt {
        fn ok(id: ProviderId, caps: ProviderCapabilities) -> Arc<Self> {
            Arc::new(FakeStt {
                id,
                caps,
                result: Mutex::new(Some(Ok(Transcript::new(id, "ok")))),
                calls: AtomicUsize::new(0),
                seen_ext: Mutex::new(None),
            })
        }

        fn err(id: ProviderId, caps: ProviderCapabilities, error: CoreError) -> Arc<Self> {
            Arc::new(FakeStt {
                id,
                caps,
                result: Mutex::new(Some(Err(error))),
                calls: AtomicUsize::new(0),
                seen_ext: Mutex::new(None),
            })
        }
    }

    impl Provider for FakeStt {
        fn id(&self) -> ProviderId {
            self.id
        }
        fn capabilities(&self) -> ProviderCapabilities {
            self.caps.clone()
        }
    }

    #[async_trait]
    impl BatchTranscriber for FakeStt {
        async fn transcribe(&self, request: TranscribeRequest) -> CoreResult<Transcript> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            *self.seen_ext.lock().unwrap() = request.source.extension();
            self.result.lock().unwrap().take().unwrap()
        }
    }

    struct FakeDecoder {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl AudioDecoder for FakeDecoder {
        async fn decode_to_wav(&self, _input: &Path, _spec: AudioSpec) -> CoreResult<PathBuf> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let f = tempfile::Builder::new().suffix(".wav").tempfile().unwrap();
            let (_, path) = f.keep().unwrap();
            std::fs::write(&path, b"RIFFfake").unwrap();
            Ok(path)
        }
        async fn decode_to_pcm(&self, _input: &Path, _spec: AudioSpec) -> CoreResult<Vec<u8>> {
            Ok(vec![0u8; 4])
        }
        async fn probe_duration(&self, _input: &Path) -> CoreResult<Option<f64>> {
            Ok(Some(1.0))
        }
    }

    fn stt_caps() -> ProviderCapabilities {
        ProviderCapabilities {
            batch_stt: true,
            ..Default::default()
        }
    }

    fn ogg_request(dir: &Path) -> TranscribeRequest {
        let path = dir.join("clip.ogg");
        std::fs::write(&path, b"OggS-fake").unwrap();
        TranscribeRequest::new(AudioSource::File(path))
    }

    #[tokio::test]
    async fn auto_falls_back_on_provider_error() {
        let failing = FakeStt::err(
            ProviderId::Xai,
            stt_caps(),
            CoreError::Auth {
                provider: "xai".into(),
                message: "bad key".into(),
            },
        );
        let working = FakeStt::ok(ProviderId::Cartesia, stt_caps());
        let engine = Engine::builder()
            .stt(failing.clone())
            .stt(working.clone())
            .build();

        let dir = tempfile::tempdir().unwrap();
        let transcript = engine
            .transcribe(ogg_request(dir.path()), None)
            .await
            .unwrap();
        assert_eq!(transcript.provider, ProviderId::Cartesia);
        assert_eq!(failing.calls.load(Ordering::SeqCst), 1);
        assert_eq!(working.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn invalid_input_does_not_fall_back() {
        let failing = FakeStt::err(
            ProviderId::Xai,
            stt_caps(),
            CoreError::InvalidInput("broken".into()),
        );
        let working = FakeStt::ok(ProviderId::Cartesia, stt_caps());
        let engine = Engine::builder().stt(failing).stt(working.clone()).build();

        let dir = tempfile::tempdir().unwrap();
        let err = engine
            .transcribe(ogg_request(dir.path()), None)
            .await
            .unwrap_err();
        assert!(matches!(err, CoreError::InvalidInput(_)));
        assert_eq!(working.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn transcodes_for_providers_that_reject_the_container() {
        let mut caps = stt_caps();
        caps.stt_input_extensions = vec!["wav".into(), "mp3".into()];
        let provider = FakeStt::ok(ProviderId::Openai, caps);
        let decoder = Arc::new(FakeDecoder {
            calls: AtomicUsize::new(0),
        });
        let engine = Engine::builder()
            .stt(provider.clone())
            .decoder(decoder.clone())
            .build();

        let dir = tempfile::tempdir().unwrap();
        engine
            .transcribe(ogg_request(dir.path()), Some(ProviderId::Openai))
            .await
            .unwrap();
        assert_eq!(decoder.calls.load(Ordering::SeqCst), 1);
        // The provider received the transcoded wav, not the ogg.
        assert_eq!(provider.seen_ext.lock().unwrap().as_deref(), Some("wav"));
    }

    #[tokio::test]
    async fn capability_mismatch_skips_provider_in_auto() {
        let mut no_diarize = stt_caps();
        no_diarize.diarization = false;
        let mut with_diarize = stt_caps();
        with_diarize.diarization = true;
        let first = FakeStt::ok(ProviderId::Cartesia, no_diarize);
        let second = FakeStt::ok(ProviderId::Xai, with_diarize);
        let engine = Engine::builder()
            .stt(first.clone())
            .stt(second.clone())
            .build();

        let dir = tempfile::tempdir().unwrap();
        let mut request = ogg_request(dir.path());
        request.diarize = true;
        let transcript = engine.transcribe(request, None).await.unwrap();
        assert_eq!(transcript.provider, ProviderId::Xai);
        assert_eq!(first.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn explicit_provider_must_be_registered() {
        let engine = Engine::builder().build();
        let dir = tempfile::tempdir().unwrap();
        let err = engine
            .transcribe(ogg_request(dir.path()), Some(ProviderId::Xai))
            .await
            .unwrap_err();
        assert!(matches!(err, CoreError::NotConfigured { .. }));
    }

    struct FakeTts {
        id: ProviderId,
        fail: bool,
    }

    impl Provider for FakeTts {
        fn id(&self) -> ProviderId {
            self.id
        }
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                batch_tts: true,
                ..Default::default()
            }
        }
    }

    #[async_trait]
    impl BatchSpeechSynthesizer for FakeTts {
        async fn synthesize(&self, request: SpeechRequest) -> CoreResult<AudioOutput> {
            if self.fail {
                return Err(CoreError::RateLimited {
                    provider: self.id.to_string(),
                    message: "busy".into(),
                });
            }
            Ok(AudioOutput {
                bytes: request.text.into_bytes(),
                mime: "audio/mpeg".into(),
                codec: request.codec,
                provider: self.id,
                duration: None,
            })
        }
    }

    #[tokio::test]
    async fn speak_auto_falls_back() {
        let engine = Engine::builder()
            .tts(Arc::new(FakeTts {
                id: ProviderId::Xai,
                fail: true,
            }))
            .tts(Arc::new(FakeTts {
                id: ProviderId::Openai,
                fail: false,
            }))
            .build();
        let output = engine
            .speak(SpeechRequest::new("hola"), None)
            .await
            .unwrap();
        assert_eq!(output.provider, ProviderId::Openai);
        assert_eq!(output.bytes, b"hola");
    }
}
