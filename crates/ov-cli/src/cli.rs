//! Clap surface + command handlers.

use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{bail, Context};
use clap::{Args, Parser, Subcommand};
use futures_util::StreamExt;
use ov_config::Config;
use ov_core::domain::{AudioCodec, AudioSource, Language, SpeechRequest, TranscribeRequest};
use ov_core::ports::{
    AudioDecoder, AudioEvent, AudioSpec, PcmChunk, Provider as _, StreamTranscribeRequest,
    TranscriptEvent,
};
use ov_core::ProviderId;
use ov_output::OutputFormat;

use crate::compose;

#[derive(Parser)]
#[command(
    name = "openvoice",
    version,
    about = "Agnostic speech-to-text and text-to-speech: local Rust engines + OpenAI, ElevenLabs, Cartesia, and xAI"
)]
pub struct Cli {
    /// Verbose logging to stderr.
    #[arg(long, global = true)]
    pub verbose: bool,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Transcribe audio files to txt/srt/vtt/json.
    Transcribe(TranscribeArgs),
    /// Synthesize speech from text.
    Speak(SpeakArgs),
    /// Realtime streaming (WebSocket providers).
    Stream {
        #[command(subcommand)]
        command: StreamCommand,
    },
    /// Inspect configured providers.
    Providers {
        #[command(subcommand)]
        command: ProvidersCommand,
    },
    /// Manage local models.
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },
}

#[derive(Args)]
pub struct TranscribeArgs {
    /// Input audio files.
    #[arg(required = true)]
    pub files: Vec<PathBuf>,
    /// Source language hint (ISO 639-1, e.g. es).
    #[arg(long)]
    pub lang: Option<String>,
    /// Provider: auto, local-canary, openai, elevenlabs, cartesia, xai.
    #[arg(long)]
    pub provider: Option<String>,
    /// Provider-specific model override.
    #[arg(long)]
    pub model: Option<String>,
    /// Comma-separated output formats: txt,srt,vtt,json.
    #[arg(long)]
    pub format: Option<String>,
    /// Write outputs here instead of next to each input.
    #[arg(long)]
    pub output_dir: Option<PathBuf>,
    /// Enable speaker diarization (providers that support it).
    #[arg(long)]
    pub diarize: bool,
    /// Bias transcription toward a term (repeatable).
    #[arg(long = "keyterm")]
    pub keyterms: Vec<String>,
    /// Context prompt (providers that support it).
    #[arg(long)]
    pub prompt: Option<String>,
}

#[derive(Args)]
pub struct SpeakArgs {
    /// Text to speak (or use --file).
    pub text: Option<String>,
    /// Read the text from a file instead.
    #[arg(long, conflicts_with = "text")]
    pub file: Option<PathBuf>,
    /// Language (BCP-47, e.g. es, en, pt-BR).
    #[arg(long)]
    pub lang: Option<String>,
    /// Voice id/name (provider-specific).
    #[arg(long)]
    pub voice: Option<String>,
    /// Provider: auto, openai, elevenlabs, cartesia, xai.
    #[arg(long)]
    pub provider: Option<String>,
    /// Provider-specific model override.
    #[arg(long)]
    pub model: Option<String>,
    /// Output audio file (extension defaults from --codec).
    #[arg(long, short)]
    pub out: Option<PathBuf>,
    /// Output codec: mp3, wav, pcm, opus, flac, aac.
    #[arg(long, default_value = "mp3")]
    pub codec: String,
    /// Output sample rate in Hz.
    #[arg(long)]
    pub sample_rate: Option<u32>,
    /// Speech speed multiplier.
    #[arg(long)]
    pub speed: Option<f32>,
    /// Style instructions (providers that support it).
    #[arg(long)]
    pub instructions: Option<String>,
}

#[derive(Subcommand)]
pub enum StreamCommand {
    /// Stream a file through realtime STT, printing partials.
    Stt(StreamSttArgs),
    /// Stream TTS audio to a file as it is generated.
    Tts(StreamTtsArgs),
}

#[derive(Args)]
pub struct StreamSttArgs {
    /// Input audio file (decoded to 16kHz PCM and streamed).
    pub file: PathBuf,
    #[arg(long)]
    pub lang: Option<String>,
    /// Provider (currently: xai).
    #[arg(long)]
    pub provider: Option<String>,
    #[arg(long)]
    pub diarize: bool,
    #[arg(long = "keyterm")]
    pub keyterms: Vec<String>,
    /// Also emit interim (non-final) hypotheses.
    #[arg(long)]
    pub interim: bool,
    /// Write final transcript outputs (txt,srt,vtt,json).
    #[arg(long)]
    pub format: Option<String>,
    #[arg(long)]
    pub output_dir: Option<PathBuf>,
}

#[derive(Args)]
pub struct StreamTtsArgs {
    /// Text to speak.
    pub text: String,
    #[arg(long)]
    pub lang: Option<String>,
    #[arg(long)]
    pub voice: Option<String>,
    /// Provider (currently: xai).
    #[arg(long)]
    pub provider: Option<String>,
    /// Output audio file.
    #[arg(long, short, default_value = "speech.mp3")]
    pub out: PathBuf,
    /// Output codec: mp3, wav, pcm.
    #[arg(long, default_value = "mp3")]
    pub codec: String,
    #[arg(long)]
    pub sample_rate: Option<u32>,
}

#[derive(Subcommand)]
pub enum ProvidersCommand {
    /// List providers and their status.
    List,
    /// Diagnose configuration problems.
    Doctor,
}

#[derive(Subcommand)]
pub enum ModelsCommand {
    /// List local models and their install state.
    List,
    /// Download a local model.
    Fetch { name: String },
    /// Delete a local model.
    Remove { name: String },
}

fn parse_provider(value: Option<&str>, config_default: &str) -> anyhow::Result<Option<ProviderId>> {
    let raw = value.unwrap_or(config_default).trim();
    if raw.is_empty() || raw.eq_ignore_ascii_case("auto") {
        return Ok(None);
    }
    ProviderId::from_str(raw)
        .map(Some)
        .map_err(anyhow::Error::msg)
}

fn parse_language(cli: Option<&str>, config_default: &str) -> Option<Language> {
    cli.map(str::to_string)
        .or_else(|| {
            if config_default.is_empty() {
                None
            } else {
                Some(config_default.to_string())
            }
        })
        .map(Language::new)
}

pub async fn run(args: Cli) -> anyhow::Result<()> {
    let config = Config::load().map_err(|e| anyhow::anyhow!(e.to_string()))?;
    match args.command {
        Command::Transcribe(cmd) => transcribe(&config, cmd).await,
        Command::Speak(cmd) => speak(&config, cmd).await,
        Command::Stream { command } => match command {
            StreamCommand::Stt(cmd) => stream_stt(&config, cmd).await,
            StreamCommand::Tts(cmd) => stream_tts(&config, cmd).await,
        },
        Command::Providers { command } => match command {
            ProvidersCommand::List => providers_list(&config),
            ProvidersCommand::Doctor => providers_doctor(&config),
        },
        Command::Models { command } => match command {
            ModelsCommand::List => models_list(&config),
            ModelsCommand::Fetch { name } => models_fetch(&config, &name).await,
            ModelsCommand::Remove { name } => models_remove(&config, &name),
        },
    }
}

async fn transcribe(config: &Config, args: TranscribeArgs) -> anyhow::Result<()> {
    let composition = compose::build(config);
    let want = parse_provider(args.provider.as_deref(), &config.defaults.stt_provider)?;
    let language = parse_language(args.lang.as_deref(), &config.defaults.language);
    let formats =
        OutputFormat::parse_list(args.format.as_deref().unwrap_or(&config.defaults.formats))
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let mut failures = 0usize;
    for file in &args.files {
        if !file.is_file() {
            eprintln!("warn: cannot read '{}', skipping", file.display());
            failures += 1;
            continue;
        }
        let mut request = TranscribeRequest::new(AudioSource::File(file.clone()));
        request.language = language.clone();
        request.model = args.model.clone();
        request.diarize = args.diarize;
        request.keyterms = args.keyterms.clone();
        request.prompt = args.prompt.clone();

        match composition.engine.transcribe(request, want).await {
            Ok(transcript) => {
                let written =
                    ov_output::write_all(&transcript, file, args.output_dir.as_deref(), &formats)
                        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                for path in written {
                    println!("{}", path.display());
                }
            }
            Err(e) => {
                eprintln!("error: {}: {e}", file.display());
                failures += 1;
            }
        }
    }
    if failures > 0 {
        bail!("{failures} file(s) failed");
    }
    Ok(())
}

async fn speak(config: &Config, args: SpeakArgs) -> anyhow::Result<()> {
    let text = match (&args.text, &args.file) {
        (Some(text), _) => text.clone(),
        (None, Some(path)) => {
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?
        }
        (None, None) => bail!("provide TEXT or --file"),
    };
    let composition = compose::build(config);
    let want = parse_provider(args.provider.as_deref(), &config.defaults.tts_provider)?;
    let codec = AudioCodec::from_str(&args.codec).map_err(anyhow::Error::msg)?;

    let mut request = SpeechRequest::new(text);
    request.language = parse_language(args.lang.as_deref(), &config.defaults.language);
    request.voice = args.voice.clone();
    request.model = args.model.clone();
    request.codec = codec;
    request.sample_rate = args.sample_rate;
    request.speed = args.speed;
    request.instructions = args.instructions.clone();

    let output = composition
        .engine
        .speak(request, want)
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let out = args
        .out
        .unwrap_or_else(|| PathBuf::from(format!("speech.{}", codec.extension())));
    std::fs::write(&out, &output.bytes).with_context(|| format!("writing {}", out.display()))?;
    eprintln!(
        "wrote {} ({} bytes, {}, via {})",
        out.display(),
        output.bytes.len(),
        output.mime,
        output.provider
    );
    println!("{}", out.display());
    Ok(())
}

async fn stream_stt(config: &Config, args: StreamSttArgs) -> anyhow::Result<()> {
    let composition = compose::build(config);
    let want = parse_provider(args.provider.as_deref(), "xai")?;
    let provider = composition
        .engine
        .streaming_stt_provider(want)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    eprintln!("streaming via {}", provider.id());

    // Decode the file to raw 16kHz mono PCM and stream it in ~100ms chunks.
    let pcm = composition
        .decoder
        .decode_to_pcm(&args.file, AudioSpec::STT_16K_MONO)
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    const CHUNK: usize = 3200; // 100ms of s16le @ 16kHz mono
    let chunks: Vec<PcmChunk> = pcm.chunks(CHUNK).map(|c| PcmChunk(c.to_vec())).collect();
    let audio: ov_core::ports::PcmStream = Box::pin(futures::stream::iter(chunks));

    let request = StreamTranscribeRequest {
        audio,
        sample_rate: 16_000,
        language: parse_language(args.lang.as_deref(), &config.defaults.language),
        diarize: args.diarize,
        keyterms: args.keyterms.clone(),
        interim_results: args.interim,
    };

    let mut stream = provider
        .stream_transcribe(request)
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let mut final_transcript = None;
    while let Some(event) = stream.next().await {
        match event.map_err(|e| anyhow::anyhow!(e.to_string()))? {
            TranscriptEvent::Partial { text, is_final, .. } => {
                let tag = if is_final { "final" } else { "partial" };
                eprintln!("[{tag}] {text}");
            }
            TranscriptEvent::Done(transcript) => {
                final_transcript = Some(transcript);
            }
        }
    }

    let transcript = final_transcript.context("stream ended without a final transcript")?;
    println!("{}", transcript.text);
    if let Some(formats) = &args.format {
        let formats =
            OutputFormat::parse_list(formats).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let written = ov_output::write_all(
            &transcript,
            &args.file,
            args.output_dir.as_deref(),
            &formats,
        )
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        for path in written {
            eprintln!("wrote {}", path.display());
        }
    }
    Ok(())
}

async fn stream_tts(config: &Config, args: StreamTtsArgs) -> anyhow::Result<()> {
    let composition = compose::build(config);
    let want = parse_provider(args.provider.as_deref(), "xai")?;
    let provider = composition
        .engine
        .streaming_tts_provider(want)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    eprintln!("streaming via {}", provider.id());

    let mut request = SpeechRequest::new(args.text.clone());
    request.language = parse_language(args.lang.as_deref(), &config.defaults.language);
    request.voice = args.voice.clone();
    request.codec = AudioCodec::from_str(&args.codec).map_err(anyhow::Error::msg)?;
    request.sample_rate = args.sample_rate;

    let mut stream = provider
        .stream_synthesize(request)
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let mut bytes: Vec<u8> = Vec::new();
    while let Some(event) = stream.next().await {
        match event.map_err(|e| anyhow::anyhow!(e.to_string()))? {
            AudioEvent::Chunk(chunk) => bytes.extend_from_slice(&chunk),
            AudioEvent::Done => break,
        }
    }
    std::fs::write(&args.out, &bytes).with_context(|| format!("writing {}", args.out.display()))?;
    eprintln!("wrote {} ({} bytes)", args.out.display(), bytes.len());
    println!("{}", args.out.display());
    Ok(())
}

fn providers_list(config: &Config) -> anyhow::Result<()> {
    let composition = compose::build(config);
    println!("{:<14} {:<12} CAPABILITIES", "PROVIDER", "STATUS");
    for id in ProviderId::ALL {
        let configured = composition.configured.contains(&id);
        let status = if configured {
            "ready"
        } else if id == ProviderId::LocalCanary {
            if !compose::local_canary_compiled() {
                "not compiled"
            } else {
                "no model"
            }
        } else {
            "no api key"
        };
        let caps = capability_summary(config, id);
        println!("{:<14} {:<12} {}", id.to_string(), status, caps);
    }
    Ok(())
}

fn capability_summary(config: &Config, id: ProviderId) -> String {
    // Capability summaries come from the adapters themselves where possible.
    let caps = match id {
        ProviderId::LocalCanary => {
            return "stt (local, 25 languages, offline)".to_string();
        }
        ProviderId::Openai => ov_providers::OpenAiProvider::new(
            "-",
            ov_providers::OpenAiSettings {
                base_url: config.providers.openai.base_url.clone(),
                ..Default::default()
            },
        )
        .capabilities(),
        ProviderId::Elevenlabs => {
            ov_providers::ElevenLabsProvider::new("-", Default::default()).capabilities()
        }
        ProviderId::Cartesia => {
            ov_providers::CartesiaProvider::new("-", Default::default()).capabilities()
        }
        ProviderId::Xai => ov_providers::XaiProvider::new("-", Default::default()).capabilities(),
    };
    let mut parts = Vec::new();
    if caps.batch_stt {
        parts.push("stt");
    }
    if caps.batch_tts {
        parts.push("tts");
    }
    if caps.streaming_stt || caps.streaming_tts {
        parts.push("streaming");
    }
    if caps.diarization {
        parts.push("diarization");
    }
    if caps.word_timestamps {
        parts.push("word-timestamps");
    }
    if caps.keyterms {
        parts.push("keyterms");
    }
    parts.join(", ")
}

fn providers_doctor(config: &Config) -> anyhow::Result<()> {
    let decoder = ov_audio::FfmpegDecoder::default();
    println!(
        "ffmpeg: {}",
        if decoder.is_available() {
            "ok"
        } else {
            "MISSING (transcoding and local/streaming decode will fail)"
        }
    );
    println!();

    for id in ProviderId::ALL {
        match id {
            ProviderId::LocalCanary => {
                let models_dir = config.models_dir();
                let installed = compose::local_canary_installed(config);
                println!("local-canary:");
                println!("  compiled: {}", compose::local_canary_compiled());
                println!("  models dir: {}", models_dir.display());
                println!(
                    "  model: {}",
                    if installed {
                        "installed".to_string()
                    } else {
                        "missing (run: openvoice models fetch canary-1b-v2)".to_string()
                    }
                );
            }
            _ => {
                let (env_name, key) = match id {
                    ProviderId::Openai => (
                        config.providers.openai.api_key_env.clone(),
                        config.api_key(id),
                    ),
                    ProviderId::Elevenlabs => (
                        config.providers.elevenlabs.api_key_env.clone(),
                        config.api_key(id),
                    ),
                    ProviderId::Cartesia => (
                        config.providers.cartesia.api_key_env.clone(),
                        config.api_key(id),
                    ),
                    ProviderId::Xai => {
                        (config.providers.xai.api_key_env.clone(), config.api_key(id))
                    }
                    ProviderId::LocalCanary => unreachable!(),
                };
                println!("{id}:");
                println!(
                    "  api key: {}",
                    if key.is_some() {
                        "present".to_string()
                    } else {
                        format!("missing (export {env_name}=...)")
                    }
                );
            }
        }
    }
    Ok(())
}

fn models_list(config: &Config) -> anyhow::Result<()> {
    let models_dir = config.models_dir();
    println!("models dir: {}", models_dir.display());
    for model in ov_local::models::ALL_MODELS {
        let installed = model.is_installed(&models_dir);
        println!(
            "{:<16} {:<12} {}",
            model.name,
            if installed { "installed" } else { "missing" },
            model.description
        );
    }
    Ok(())
}

async fn models_fetch(config: &Config, name: &str) -> anyhow::Result<()> {
    let model = ov_local::models::find(name)
        .with_context(|| format!("unknown model '{name}' (see: openvoice models list)"))?;
    let models_dir = config.models_dir();
    eprintln!("fetching {} into {}", model.name, models_dir.display());
    let dir = model
        .fetch(&models_dir, "https://huggingface.co", |file, bytes| {
            if bytes == 0 {
                eprintln!("  {file}: already present");
            } else {
                eprintln!("  {file}: {bytes} bytes");
            }
        })
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    println!("{}", dir.display());
    if !compose::local_canary_compiled() {
        eprintln!(
            "note: this build has no local inference (compile with --features local to use it)"
        );
    }
    Ok(())
}

fn models_remove(config: &Config, name: &str) -> anyhow::Result<()> {
    let model = ov_local::models::find(name)
        .with_context(|| format!("unknown model '{name}' (see: openvoice models list)"))?;
    model
        .remove(&config.models_dir())
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    eprintln!("removed {}", model.name);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_transcribe_invocation() {
        let cli = Cli::try_parse_from([
            "openvoice",
            "transcribe",
            "audio.ogg",
            "--lang",
            "es",
            "--provider",
            "xai",
            "--format",
            "txt,srt,json",
            "--keyterm",
            "Hola",
            "--keyterm",
            "Mundo",
        ])
        .unwrap();
        match cli.command {
            Command::Transcribe(args) => {
                assert_eq!(args.files.len(), 1);
                assert_eq!(args.lang.as_deref(), Some("es"));
                assert_eq!(args.keyterms, vec!["Hola", "Mundo"]);
            }
            _ => panic!("wrong command"),
        }
    }

    #[test]
    fn parses_speak_and_stream() {
        let cli = Cli::try_parse_from([
            "openvoice",
            "speak",
            "hola",
            "--provider",
            "xai",
            "--codec",
            "wav",
        ])
        .unwrap();
        assert!(matches!(cli.command, Command::Speak(_)));

        let cli =
            Cli::try_parse_from(["openvoice", "stream", "stt", "a.ogg", "--interim"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Stream {
                command: StreamCommand::Stt(_)
            }
        ));
    }

    #[test]
    fn provider_parsing_defaults() {
        assert_eq!(parse_provider(None, "auto").unwrap(), None);
        assert_eq!(parse_provider(None, "xai").unwrap(), Some(ProviderId::Xai));
        assert_eq!(
            parse_provider(Some("local-canary"), "auto").unwrap(),
            Some(ProviderId::LocalCanary)
        );
        assert!(parse_provider(Some("bogus"), "auto").is_err());
    }
}
