//! Normalized domain types. Every provider adapter maps its wire format into
//! these, so a transcript from local Canary and one from xAI are
//! indistinguishable to the output writers and the CLI.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::provider::ProviderId;

/// ISO 639-1/BCP-47 language code as the user supplied it (e.g. `es`, `en`,
/// `pt-BR`). Kept as a thin newtype so signatures stay self-documenting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Language(pub String);

impl Language {
    pub fn new(code: impl Into<String>) -> Self {
        Language(code.into().trim().to_string())
    }

    pub fn code(&self) -> &str {
        &self.0
    }

    /// Primary subtag, lowercased: `es-MX` -> `es`.
    pub fn primary(&self) -> String {
        self.0
            .split(['-', '_'])
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase()
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Where the audio bytes come from.
#[derive(Debug, Clone)]
pub enum AudioSource {
    File(PathBuf),
    Bytes {
        data: Vec<u8>,
        file_name: String,
        mime: String,
    },
}

impl AudioSource {
    pub fn file_name(&self) -> String {
        match self {
            AudioSource::File(p) => p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "audio".to_string()),
            AudioSource::Bytes { file_name, .. } => file_name.clone(),
        }
    }

    /// Lowercased extension of the underlying file name, if any.
    pub fn extension(&self) -> Option<String> {
        let name = self.file_name();
        let ext = std::path::Path::new(&name)
            .extension()?
            .to_string_lossy()
            .to_ascii_lowercase();
        Some(ext)
    }
}

/// A word with timing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

impl Word {
    pub fn new(text: impl Into<String>, start: f64, end: f64) -> Self {
        Word {
            text: text.into(),
            start,
            end,
            speaker: None,
            channel: None,
            confidence: None,
        }
    }
}

/// A subtitle-sized span of speech.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel: Option<u32>,
}

impl Segment {
    pub fn new(start: f64, end: f64, text: impl Into<String>) -> Self {
        Segment {
            start,
            end,
            text: text.into(),
            speaker: None,
            channel: None,
        }
    }
}

/// The normalized transcription result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcript {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
    pub provider: ProviderId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default)]
    pub segments: Vec<Segment>,
    #[serde(default)]
    pub words: Vec<Word>,
}

impl Transcript {
    pub fn new(provider: ProviderId, text: impl Into<String>) -> Self {
        Transcript {
            text: text.into(),
            language: None,
            duration: None,
            provider,
            model: None,
            segments: Vec::new(),
            words: Vec::new(),
        }
    }

    /// Segments to render subtitles from: the provider's own segments when
    /// present, otherwise segments synthesized from word timings, otherwise a
    /// single full-length segment.
    pub fn render_segments(&self) -> Vec<Segment> {
        if !self.segments.is_empty() {
            return self.segments.clone();
        }
        if !self.words.is_empty() {
            return segments_from_words(&self.words);
        }
        if self.text.trim().is_empty() {
            return Vec::new();
        }
        vec![Segment::new(
            0.0,
            self.duration.unwrap_or(0.0),
            self.text.trim(),
        )]
    }
}

/// Group word timings into subtitle-sized segments. Splits on sentence-final
/// punctuation, long silences (> 1.0s between words), speaker changes, or a
/// 200-character cap — the same heuristics the previous Python stack used.
pub fn segments_from_words(words: &[Word]) -> Vec<Segment> {
    const MAX_CHARS: usize = 200;
    const MAX_GAP: f64 = 1.0;

    let mut segments = Vec::new();
    let mut buf: Vec<&Word> = Vec::new();

    let flush = |buf: &mut Vec<&Word>, segments: &mut Vec<Segment>| {
        if buf.is_empty() {
            return;
        }
        let text = buf
            .iter()
            .map(|w| w.text.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        if !text.is_empty() {
            let mut seg = Segment::new(buf[0].start, buf[buf.len() - 1].end, text);
            seg.speaker = buf[0].speaker.clone();
            seg.channel = buf[0].channel;
            segments.push(seg);
        }
        buf.clear();
    };

    for word in words {
        if let Some(prev) = buf.last() {
            let gap = word.start - prev.end;
            let speaker_changed = prev.speaker != word.speaker || prev.channel != word.channel;
            let too_long = buf.iter().map(|w| w.text.len() + 1).sum::<usize>() > MAX_CHARS;
            if gap > MAX_GAP || speaker_changed || too_long {
                flush(&mut buf, &mut segments);
            }
        }
        buf.push(word);
        if word.text.trim_end().ends_with(['.', '?', '!']) {
            flush(&mut buf, &mut segments);
        }
    }
    flush(&mut buf, &mut segments);
    segments
}

/// A batch STT request, provider-agnostic.
#[derive(Debug, Clone)]
pub struct TranscribeRequest {
    pub source: AudioSource,
    pub language: Option<Language>,
    pub model: Option<String>,
    pub diarize: bool,
    pub word_timestamps: bool,
    pub keyterms: Vec<String>,
    pub prompt: Option<String>,
}

impl TranscribeRequest {
    pub fn new(source: AudioSource) -> Self {
        TranscribeRequest {
            source,
            language: None,
            model: None,
            diarize: false,
            word_timestamps: true,
            keyterms: Vec::new(),
            prompt: None,
        }
    }
}

/// Requested audio encoding for TTS output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioCodec {
    Mp3,
    Wav,
    Pcm,
    Opus,
    Flac,
    Aac,
}

impl AudioCodec {
    pub fn as_str(&self) -> &'static str {
        match self {
            AudioCodec::Mp3 => "mp3",
            AudioCodec::Wav => "wav",
            AudioCodec::Pcm => "pcm",
            AudioCodec::Opus => "opus",
            AudioCodec::Flac => "flac",
            AudioCodec::Aac => "aac",
        }
    }

    pub fn extension(&self) -> &'static str {
        match self {
            AudioCodec::Pcm => "pcm",
            other => other.as_str(),
        }
    }
}

impl std::str::FromStr for AudioCodec {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "mp3" => Ok(AudioCodec::Mp3),
            "wav" => Ok(AudioCodec::Wav),
            "pcm" => Ok(AudioCodec::Pcm),
            "opus" => Ok(AudioCodec::Opus),
            "flac" => Ok(AudioCodec::Flac),
            "aac" => Ok(AudioCodec::Aac),
            other => Err(format!("unknown audio codec '{other}'")),
        }
    }
}

/// A batch TTS request, provider-agnostic.
#[derive(Debug, Clone)]
pub struct SpeechRequest {
    pub text: String,
    pub language: Option<Language>,
    pub voice: Option<String>,
    pub model: Option<String>,
    pub codec: AudioCodec,
    pub sample_rate: Option<u32>,
    pub speed: Option<f32>,
    /// Free-form style guidance (only some providers support it, e.g. OpenAI
    /// `instructions`).
    pub instructions: Option<String>,
}

impl SpeechRequest {
    pub fn new(text: impl Into<String>) -> Self {
        SpeechRequest {
            text: text.into(),
            language: None,
            voice: None,
            model: None,
            codec: AudioCodec::Mp3,
            sample_rate: None,
            speed: None,
            instructions: None,
        }
    }
}

/// The normalized TTS result.
#[derive(Debug, Clone)]
pub struct AudioOutput {
    pub bytes: Vec<u8>,
    pub mime: String,
    pub codec: AudioCodec,
    pub provider: ProviderId,
    pub duration: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(text: &str, start: f64, end: f64) -> Word {
        Word::new(text, start, end)
    }

    #[test]
    fn language_primary_subtag() {
        assert_eq!(Language::new("es-MX").primary(), "es");
        assert_eq!(Language::new("PT_br").primary(), "pt");
        assert_eq!(Language::new("en").primary(), "en");
    }

    #[test]
    fn segments_split_on_punctuation_and_gap() {
        let words = vec![
            w("Hola", 0.0, 0.4),
            w("mundo.", 0.5, 0.9),
            w("Segunda", 3.0, 3.4),
            w("frase", 3.5, 3.9),
        ];
        let segs = segments_from_words(&words);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].text, "Hola mundo.");
        assert!((segs[0].start - 0.0).abs() < f64::EPSILON);
        assert!((segs[0].end - 0.9).abs() < f64::EPSILON);
        assert_eq!(segs[1].text, "Segunda frase");
    }

    #[test]
    fn segments_split_on_speaker_change() {
        let mut a = w("Hola", 0.0, 0.4);
        a.speaker = Some("speaker_0".into());
        let mut b = w("adios", 0.5, 0.9);
        b.speaker = Some("speaker_1".into());
        let segs = segments_from_words(&[a, b]);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].speaker.as_deref(), Some("speaker_0"));
        assert_eq!(segs[1].speaker.as_deref(), Some("speaker_1"));
    }

    #[test]
    fn render_segments_falls_back_to_full_text() {
        let mut t = Transcript::new(ProviderId::Xai, " hola mundo ");
        t.duration = Some(2.5);
        let segs = t.render_segments();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, "hola mundo");
        assert!((segs[0].end - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn audio_source_extension() {
        let s = AudioSource::File(PathBuf::from(
            "/tmp/WhatsApp Ptt 2026-07-02 at 14.56.03.ogg",
        ));
        assert_eq!(s.extension().as_deref(), Some("ogg"));
    }
}
