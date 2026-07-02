//! # ov-output
//!
//! Transcript writers: txt, srt, vtt, json. All writers consume the
//! normalized `ov_core::domain::Transcript`, so every provider produces
//! byte-identical output shapes.
//!
//! Output naming strips only the **final** extension of the input file name
//! and never touches interior dots — `WhatsApp Ptt ... 14.56.03.ogg` becomes
//! `... 14.56.03.txt` (the old Python stack lost the `.03` here).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use ov_core::domain::{Segment, Transcript};
use ov_core::{CoreError, CoreResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutputFormat {
    Txt,
    Srt,
    Vtt,
    Json,
}

impl OutputFormat {
    pub fn extension(&self) -> &'static str {
        match self {
            OutputFormat::Txt => "txt",
            OutputFormat::Srt => "srt",
            OutputFormat::Vtt => "vtt",
            OutputFormat::Json => "json",
        }
    }

    /// Parse a comma-separated list like `txt,srt,json`, preserving order and
    /// dropping duplicates.
    pub fn parse_list(list: &str) -> CoreResult<Vec<OutputFormat>> {
        let mut formats = Vec::new();
        for item in list.split(',') {
            let item = item.trim();
            if item.is_empty() {
                continue;
            }
            let f = OutputFormat::from_str(item)?;
            if !formats.contains(&f) {
                formats.push(f);
            }
        }
        if formats.is_empty() {
            return Err(CoreError::InvalidInput(format!(
                "no output formats in '{list}'"
            )));
        }
        Ok(formats)
    }
}

impl FromStr for OutputFormat {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "txt" | "text" => Ok(OutputFormat::Txt),
            "srt" => Ok(OutputFormat::Srt),
            "vtt" => Ok(OutputFormat::Vtt),
            "json" => Ok(OutputFormat::Json),
            other => Err(CoreError::InvalidInput(format!(
                "unknown output format '{other}' (expected txt, srt, vtt, json)"
            ))),
        }
    }
}

/// Compute the output path for `input` with `extension`, honoring an optional
/// output directory. Only the final extension is replaced; interior dots in
/// the file name are preserved.
pub fn output_path(input: &Path, output_dir: Option<&Path>, extension: &str) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "transcript".to_string());
    let dir = output_dir
        .map(Path::to_path_buf)
        .or_else(|| input.parent().map(Path::to_path_buf))
        .unwrap_or_default();
    dir.join(format!("{stem}.{extension}"))
}

fn fmt_timestamp(seconds: f64, comma: bool) -> String {
    let seconds = seconds.max(0.0);
    let total_ms = (seconds * 1000.0).round() as u64;
    let h = total_ms / 3_600_000;
    let m = (total_ms % 3_600_000) / 60_000;
    let s = (total_ms % 60_000) / 1000;
    let ms = total_ms % 1000;
    let sep = if comma { ',' } else { '.' };
    format!("{h:02}:{m:02}:{s:02}{sep}{ms:03}")
}

fn speakers_present(segments: &[Segment]) -> bool {
    segments.iter().any(|s| s.speaker.is_some())
}

pub fn render_txt(transcript: &Transcript) -> String {
    let segments = transcript.render_segments();
    if !speakers_present(&segments) {
        let text = transcript.text.trim();
        if !text.is_empty() {
            return format!("{text}\n");
        }
        let joined = segments
            .iter()
            .map(|s| s.text.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        return format!("{joined}\n");
    }
    // Speaker-labelled paragraphs: new paragraph on speaker change.
    let mut out = String::new();
    let mut last: Option<&str> = None;
    for seg in &segments {
        let speaker = seg.speaker.as_deref().unwrap_or("unknown");
        if last != Some(speaker) {
            if !out.is_empty() {
                out.push_str("\n\n");
            }
            let _ = write!(out, "[{speaker}] ");
            last = Some(speaker);
        } else {
            out.push(' ');
        }
        out.push_str(seg.text.trim());
    }
    out.push('\n');
    out
}

pub fn render_srt(transcript: &Transcript) -> String {
    let segments = transcript.render_segments();
    let with_speakers = speakers_present(&segments);
    let mut out = String::new();
    for (i, seg) in segments
        .iter()
        .filter(|s| !s.text.trim().is_empty())
        .enumerate()
    {
        let start = fmt_timestamp(seg.start, true);
        let end = fmt_timestamp(seg.end, true);
        let mut text = seg.text.trim().to_string();
        if with_speakers {
            if let Some(speaker) = &seg.speaker {
                text = format!("[{speaker}] {text}");
            }
        }
        let _ = write!(out, "{}\n{start} --> {end}\n{text}\n\n", i + 1);
    }
    out.trim_end_matches('\n').to_string() + "\n"
}

pub fn render_vtt(transcript: &Transcript) -> String {
    let segments = transcript.render_segments();
    let with_speakers = speakers_present(&segments);
    let mut out = String::from("WEBVTT\n\n");
    for seg in segments.iter().filter(|s| !s.text.trim().is_empty()) {
        let start = fmt_timestamp(seg.start, false);
        let end = fmt_timestamp(seg.end, false);
        let mut text = seg.text.trim().to_string();
        if with_speakers {
            if let Some(speaker) = &seg.speaker {
                text = format!("<v {speaker}>{text}");
            }
        }
        let _ = write!(out, "{start} --> {end}\n{text}\n\n");
    }
    out.trim_end_matches('\n').to_string() + "\n"
}

pub fn render_json(transcript: &Transcript) -> CoreResult<String> {
    let mut value = serde_json::to_value(transcript)
        .map_err(|e| CoreError::Io(format!("serializing transcript: {e}")))?;
    // Always include the rendered segments so downstream consumers never need
    // to re-implement the word-grouping heuristics.
    if transcript.segments.is_empty() {
        let rendered = transcript.render_segments();
        if let Some(obj) = value.as_object_mut() {
            obj.insert(
                "segments".to_string(),
                serde_json::to_value(rendered)
                    .map_err(|e| CoreError::Io(format!("serializing segments: {e}")))?,
            );
        }
    }
    serde_json::to_string_pretty(&value)
        .map(|s| s + "\n")
        .map_err(|e| CoreError::Io(format!("serializing transcript: {e}")))
}

/// Write `transcript` next to `input` (or into `output_dir`) in each of the
/// requested formats. Returns the written paths in request order.
pub fn write_all(
    transcript: &Transcript,
    input: &Path,
    output_dir: Option<&Path>,
    formats: &[OutputFormat],
) -> CoreResult<Vec<PathBuf>> {
    let mut written = Vec::new();
    for format in formats {
        let path = output_path(input, output_dir, format.extension());
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| CoreError::Io(format!("creating {}: {e}", parent.display())))?;
            }
        }
        let content = match format {
            OutputFormat::Txt => render_txt(transcript),
            OutputFormat::Srt => render_srt(transcript),
            OutputFormat::Vtt => render_vtt(transcript),
            OutputFormat::Json => render_json(transcript)?,
        };
        std::fs::write(&path, content)
            .map_err(|e| CoreError::Io(format!("writing {}: {e}", path.display())))?;
        written.push(path);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ov_core::domain::{Segment, Transcript, Word};
    use ov_core::ProviderId;

    fn transcript() -> Transcript {
        let mut t = Transcript::new(ProviderId::Xai, "Hola mundo. Segunda frase");
        t.language = Some("es".into());
        t.duration = Some(4.0);
        t.segments = vec![
            Segment::new(0.0, 0.9, "Hola mundo."),
            Segment::new(3.0, 3.9, "Segunda frase"),
        ];
        t
    }

    #[test]
    fn output_path_preserves_interior_dots() {
        let input = Path::new("/downloads/WhatsApp Ptt 2026-07-02 at 14.56.03.ogg");
        let out = output_path(input, None, "txt");
        assert_eq!(
            out,
            PathBuf::from("/downloads/WhatsApp Ptt 2026-07-02 at 14.56.03.txt")
        );
        let out = output_path(input, Some(Path::new("/tmp/out")), "srt");
        assert_eq!(
            out,
            PathBuf::from("/tmp/out/WhatsApp Ptt 2026-07-02 at 14.56.03.srt")
        );
    }

    #[test]
    fn srt_renders_expected_shape() {
        let srt = render_srt(&transcript());
        assert_eq!(
            srt,
            "1\n00:00:00,000 --> 00:00:00,900\nHola mundo.\n\n2\n00:00:03,000 --> 00:00:03,900\nSegunda frase\n"
        );
    }

    #[test]
    fn vtt_renders_expected_shape() {
        let vtt = render_vtt(&transcript());
        assert!(vtt.starts_with("WEBVTT\n\n00:00:00.000 --> 00:00:00.900\nHola mundo.\n"));
    }

    #[test]
    fn txt_prefers_full_text() {
        assert_eq!(render_txt(&transcript()), "Hola mundo. Segunda frase\n");
    }

    #[test]
    fn txt_labels_speakers() {
        let mut t = transcript();
        t.segments[0].speaker = Some("speaker_0".into());
        t.segments[1].speaker = Some("speaker_1".into());
        let txt = render_txt(&t);
        assert_eq!(
            txt,
            "[speaker_0] Hola mundo.\n\n[speaker_1] Segunda frase\n"
        );
    }

    #[test]
    fn srt_from_words_when_no_segments() {
        let mut t = Transcript::new(ProviderId::Cartesia, "Hola mundo.");
        t.words = vec![Word::new("Hola", 0.0, 0.4), Word::new("mundo.", 0.5, 0.9)];
        let srt = render_srt(&t);
        assert!(srt.contains("Hola mundo."));
        assert!(srt.contains("00:00:00,000 --> 00:00:00,900"));
    }

    #[test]
    fn json_includes_rendered_segments() {
        let mut t = Transcript::new(ProviderId::Openai, "Hola mundo.");
        t.words = vec![Word::new("Hola", 0.0, 0.4), Word::new("mundo.", 0.5, 0.9)];
        let json = render_json(&t).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["provider"], "openai");
        assert_eq!(value["segments"][0]["text"], "Hola mundo.");
    }

    #[test]
    fn timestamp_rounding() {
        assert_eq!(fmt_timestamp(3599.9995, true), "01:00:00,000");
        assert_eq!(fmt_timestamp(0.001, false), "00:00:00.001");
        assert_eq!(fmt_timestamp(-1.0, true), "00:00:00,000");
    }

    #[test]
    fn write_all_creates_files() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("a.b.c.ogg");
        let formats = OutputFormat::parse_list("txt,srt,json").unwrap();
        let written = write_all(&transcript(), &input, None, &formats).unwrap();
        assert_eq!(written.len(), 3);
        assert_eq!(written[0], dir.path().join("a.b.c.txt"));
        assert!(written.iter().all(|p| p.exists()));
    }

    #[test]
    fn parse_list_dedupes_and_validates() {
        let formats = OutputFormat::parse_list("txt, srt,txt").unwrap();
        assert_eq!(formats, vec![OutputFormat::Txt, OutputFormat::Srt]);
        assert!(OutputFormat::parse_list("nope").is_err());
        assert!(OutputFormat::parse_list("").is_err());
    }
}
