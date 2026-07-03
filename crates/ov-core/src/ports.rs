//! Port traits (hexagonal architecture). Kept deliberately narrow (ISP): a
//! provider implements only the capabilities it actually has, and the engine
//! depends on these traits — never on a concrete adapter (DIP).

use std::path::{Path, PathBuf};
use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::capabilities::ProviderCapabilities;
use crate::domain::{AudioOutput, SpeechRequest, TranscribeRequest, Transcript, Word};
use crate::error::CoreResult;
use crate::provider::ProviderId;

/// Identity + advertised capabilities. Implemented by every adapter.
pub trait Provider: Send + Sync {
    fn id(&self) -> ProviderId;
    fn capabilities(&self) -> ProviderCapabilities;
}

/// Batch speech-to-text.
#[async_trait]
pub trait BatchTranscriber: Provider {
    async fn transcribe(&self, request: TranscribeRequest) -> CoreResult<Transcript>;
}

/// Batch text-to-speech.
#[async_trait]
pub trait BatchSpeechSynthesizer: Provider {
    async fn synthesize(&self, request: SpeechRequest) -> CoreResult<AudioOutput>;
}

/// An incremental transcription event from a streaming session.
#[derive(Debug, Clone)]
pub enum TranscriptEvent {
    /// Interim or chunk-final text. `is_final` locks the text; `speech_final`
    /// marks the end of an utterance.
    Partial {
        text: String,
        words: Vec<Word>,
        is_final: bool,
        speech_final: bool,
    },
    /// The stitched, final transcript. The stream ends after this.
    Done(Transcript),
}

pub type TranscriptStream = Pin<Box<dyn Stream<Item = CoreResult<TranscriptEvent>> + Send>>;

/// Raw PCM audio pushed into a streaming STT session.
#[derive(Debug, Clone)]
pub struct PcmChunk(pub Vec<u8>);

pub type PcmStream = Pin<Box<dyn Stream<Item = PcmChunk> + Send>>;

/// Streaming STT session parameters (raw PCM in, transcript events out).
pub struct StreamTranscribeRequest {
    pub audio: PcmStream,
    pub sample_rate: u32,
    pub language: Option<crate::domain::Language>,
    pub diarize: bool,
    pub keyterms: Vec<String>,
    pub interim_results: bool,
}

/// Streaming speech-to-text.
#[async_trait]
pub trait StreamingTranscriber: Provider {
    async fn stream_transcribe(
        &self,
        request: StreamTranscribeRequest,
    ) -> CoreResult<TranscriptStream>;
}

/// An audio chunk from a streaming TTS session.
#[derive(Debug, Clone)]
pub enum AudioEvent {
    Chunk(Vec<u8>),
    Done,
}

pub type AudioEventStream = Pin<Box<dyn Stream<Item = CoreResult<AudioEvent>> + Send>>;

/// Streaming text-to-speech.
#[async_trait]
pub trait StreamingSpeechSynthesizer: Provider {
    async fn stream_synthesize(&self, request: SpeechRequest) -> CoreResult<AudioEventStream>;
}

/// Target PCM/WAV spec for local decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioSpec {
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioSpec {
    /// The spec every local STT engine consumes: 16 kHz mono.
    pub const STT_16K_MONO: AudioSpec = AudioSpec {
        sample_rate: 16_000,
        channels: 1,
    };
}

/// Local audio encoding (ffmpeg adapter lives in ov-audio). Used by local
/// TTS engines that natively emit WAV to serve other output codecs.
#[async_trait]
pub trait AudioEncoder: Send + Sync {
    /// Re-encode a complete in-memory WAV file into `codec` (optionally
    /// resampling to `sample_rate`).
    async fn encode_wav(
        &self,
        wav: &[u8],
        codec: crate::domain::AudioCodec,
        sample_rate: Option<u32>,
    ) -> CoreResult<Vec<u8>>;
}

/// Local audio decoding (ffmpeg adapter lives in ov-audio).
#[async_trait]
pub trait AudioDecoder: Send + Sync {
    /// Decode any input into a 16-bit PCM WAV file at `spec`, returning the
    /// path of the produced (temporary) file.
    async fn decode_to_wav(&self, input: &Path, spec: AudioSpec) -> CoreResult<PathBuf>;

    /// Decode any input into raw interleaved s16le PCM bytes at `spec`.
    async fn decode_to_pcm(&self, input: &Path, spec: AudioSpec) -> CoreResult<Vec<u8>>;

    /// Best-effort duration probe in seconds.
    async fn probe_duration(&self, input: &Path) -> CoreResult<Option<f64>>;
}
