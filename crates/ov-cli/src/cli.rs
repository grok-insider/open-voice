//! Clap surface + command handlers.

use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{bail, Context};
use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::Shell as CompletionShell;
use futures_util::StreamExt;
use ov_config::Config;
use ov_core::domain::{AudioCodec, AudioSource, Language, SpeechRequest, TranscribeRequest};
use ov_core::ports::{
    AudioDecoder, AudioEvent, AudioSpec, PcmChunk, Provider as _, StreamTranscribeRequest,
    TranscriptEvent,
};
use ov_core::ProviderId;
use ov_output::OutputFormat;
use ov_providers::RealtimeAgentRequest;

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
    /// Run one xAI realtime Voice Agent text turn.
    Agent(AgentArgs),
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
    /// Discover and manage TTS voices.
    Voices {
        #[command(subcommand)]
        command: VoicesCommand,
    },
    /// Generate shell completions.
    Completions {
        /// Shell to generate completions for.
        shell: CompletionShell,
    },
    /// Run opt-in end-to-end smoke checks.
    Smoke {
        /// Smoke target to exercise.
        #[arg(value_enum)]
        target: SmokeTarget,
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
    /// Provider: auto, local-qwen3, openai, elevenlabs, cartesia, xai.
    #[arg(long)]
    pub provider: Option<String>,
    /// Provider-specific model override.
    #[arg(long)]
    pub model: Option<String>,
    /// Output audio file (extension defaults from --codec).
    #[arg(long, short)]
    pub out: Option<PathBuf>,
    /// Output codec: mp3, wav, pcm, mulaw, alaw, opus, flac, aac.
    #[arg(long, default_value = "mp3")]
    pub codec: String,
    /// Output sample rate in Hz.
    #[arg(long)]
    pub sample_rate: Option<u32>,
    /// Output bitrate in bits per second (xAI mp3).
    #[arg(long)]
    pub bit_rate: Option<u32>,
    /// Speech speed multiplier.
    #[arg(long)]
    pub speed: Option<f32>,
    /// Optimize streaming latency, provider-specific (xAI: 0-2).
    #[arg(long)]
    pub optimize_streaming_latency: Option<u8>,
    /// Ask provider to normalize numbers, abbreviations, and symbols.
    #[arg(long)]
    pub text_normalization: bool,
    /// Request timestamp metadata when the provider supports it.
    #[arg(long)]
    pub with_timestamps: bool,
    /// Style instructions (providers that support it).
    #[arg(long)]
    pub instructions: Option<String>,
    /// Split long text into chunks, synthesize each chunk, and stitch audio.
    #[arg(long)]
    pub long: bool,
    /// Approximate max characters per long-form chunk.
    #[arg(long, default_value_t = 1200)]
    pub chunk_chars: usize,
    /// Write long-form chunk metadata to this JSON file.
    #[arg(long)]
    pub manifest: Option<PathBuf>,
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
    /// Enable xAI Smart Turn end-of-turn detection.
    #[arg(long)]
    pub smart_turn: bool,
    /// Maximum Smart Turn wait in milliseconds.
    #[arg(long)]
    pub smart_turn_timeout_ms: Option<u64>,
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
    /// Output codec: mp3, wav, pcm, mulaw, alaw.
    #[arg(long, default_value = "mp3")]
    pub codec: String,
    #[arg(long)]
    pub sample_rate: Option<u32>,
    #[arg(long)]
    pub bit_rate: Option<u32>,
    #[arg(long)]
    pub optimize_streaming_latency: Option<u8>,
}

#[derive(Args)]
pub struct AgentArgs {
    /// User text turn to send to the realtime voice agent.
    pub text: String,
    /// xAI voice id: eve, ara, rex, sal, leo, or a custom voice_id.
    #[arg(long)]
    pub voice: Option<String>,
    /// Realtime model: grok-voice-latest, grok-voice-fast-1.0, etc.
    #[arg(long)]
    pub model: Option<String>,
    /// Session instructions.
    #[arg(long)]
    pub instructions: Option<String>,
    /// Reasoning effort: high or none.
    #[arg(long)]
    pub reasoning_effort: Option<String>,
    /// Output audio file. Realtime audio is raw PCM/G.711 bytes.
    #[arg(long, short)]
    pub out: Option<PathBuf>,
    /// Output codec: pcm, mulaw, alaw.
    #[arg(long, default_value = "pcm")]
    pub codec: String,
    /// Output audio sample rate.
    #[arg(long, default_value_t = 24_000)]
    pub sample_rate: u32,
    /// Request text only, no audio modality.
    #[arg(long)]
    pub text_only: bool,
    /// Configure manual turn mode instead of server VAD.
    #[arg(long)]
    pub manual_turn: bool,
    /// Server VAD threshold.
    #[arg(long)]
    pub vad_threshold: Option<f32>,
    /// Server VAD silence duration in milliseconds.
    #[arg(long)]
    pub vad_silence_ms: Option<u64>,
    /// Server VAD prefix padding in milliseconds.
    #[arg(long)]
    pub vad_prefix_padding_ms: Option<u64>,
    /// Enable input transcription with this language hint.
    #[arg(long)]
    pub language_hint: Option<String>,
    /// Emit the whole turn as JSON.
    #[arg(long)]
    pub json: bool,
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

#[derive(Subcommand)]
pub enum VoicesCommand {
    /// List local and xAI voices.
    List(VoicesListArgs),
    /// Get one xAI custom voice by id.
    Get { id: String },
    /// Clone a voice from a reference audio file using xAI Custom Voices.
    Clone(VoiceCloneArgs),
    /// Update xAI custom voice metadata.
    Update(VoiceUpdateArgs),
    /// Delete an xAI custom voice.
    Delete {
        id: String,
        /// Delete without an interactive confirmation prompt.
        #[arg(long)]
        yes: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum VoiceProviderFilter {
    All,
    Local,
    Xai,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SmokeTarget {
    Local,
    Xai,
}

#[derive(Args)]
pub struct VoicesListArgs {
    /// Voice source to list.
    #[arg(long, value_enum, default_value_t = VoiceProviderFilter::All)]
    pub provider: VoiceProviderFilter,
    /// Only show xAI custom voices.
    #[arg(long)]
    pub custom_only: bool,
    /// Emit JSON instead of a table.
    #[arg(long)]
    pub json: bool,
    /// xAI custom voice page size.
    #[arg(long, default_value_t = 50)]
    pub limit: u32,
    /// xAI custom voice pagination token.
    #[arg(long)]
    pub page_token: Option<String>,
}

#[derive(Args)]
pub struct VoiceCloneArgs {
    /// Reference audio file (max 120 seconds per xAI docs).
    #[arg(long)]
    pub file: PathBuf,
    /// Display name for the custom voice.
    #[arg(long)]
    pub name: String,
    #[arg(long)]
    pub description: Option<String>,
    #[arg(long)]
    pub gender: Option<String>,
    #[arg(long)]
    pub accent: Option<String>,
    #[arg(long)]
    pub age: Option<String>,
    #[arg(long)]
    pub language: Option<String>,
    #[arg(long)]
    pub use_case: Option<String>,
    #[arg(long)]
    pub tone: Option<String>,
}

#[derive(Args)]
pub struct VoiceUpdateArgs {
    pub id: String,
    #[arg(long)]
    pub name: Option<String>,
    #[arg(long)]
    pub description: Option<String>,
    #[arg(long)]
    pub gender: Option<String>,
    #[arg(long)]
    pub accent: Option<String>,
    #[arg(long)]
    pub age: Option<String>,
    #[arg(long)]
    pub language: Option<String>,
    #[arg(long)]
    pub use_case: Option<String>,
    #[arg(long)]
    pub tone: Option<String>,
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
        Command::Agent(cmd) => agent(&config, cmd).await,
        Command::Providers { command } => match command {
            ProvidersCommand::List => providers_list(&config),
            ProvidersCommand::Doctor => providers_doctor(&config),
        },
        Command::Models { command } => match command {
            ModelsCommand::List => models_list(&config),
            ModelsCommand::Fetch { name } => models_fetch(&config, &name).await,
            ModelsCommand::Remove { name } => models_remove(&config, &name),
        },
        Command::Voices { command } => voices(&config, command).await,
        Command::Completions { shell } => completions(shell),
        Command::Smoke { target } => smoke(&config, target).await,
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

    if args.long {
        return speak_long(
            &composition,
            want,
            &args,
            &config.defaults.language,
            text,
            codec,
        )
        .await;
    }

    let request = speech_request_from_args(&args, &config.defaults.language, text, codec);

    let output = composition
        .engine
        .speak(request, want)
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let out = args
        .out
        .unwrap_or_else(|| PathBuf::from(format!("speech.{}", codec.extension())));
    std::fs::write(&out, &output.bytes).with_context(|| format!("writing {}", out.display()))?;
    if let Some(metadata) = &output.metadata {
        let sidecar = out.with_extension(format!(
            "{}.json",
            out.extension()
                .and_then(|e| e.to_str())
                .unwrap_or_else(|| output.codec.extension())
        ));
        std::fs::write(&sidecar, serde_json::to_string_pretty(metadata)? + "\n")
            .with_context(|| format!("writing {}", sidecar.display()))?;
        eprintln!("wrote {} (timestamp metadata)", sidecar.display());
    }
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

fn speech_request_from_args(
    args: &SpeakArgs,
    default_language: &str,
    text: String,
    codec: AudioCodec,
) -> SpeechRequest {
    let mut request = SpeechRequest::new(text);
    request.language = parse_language(args.lang.as_deref(), default_language);
    request.voice = args.voice.clone();
    request.model = args.model.clone();
    request.codec = codec;
    request.sample_rate = args.sample_rate;
    request.bit_rate = args.bit_rate;
    request.speed = args.speed;
    request.optimize_streaming_latency = args.optimize_streaming_latency;
    request.text_normalization = args.text_normalization.then_some(true);
    request.with_timestamps = args.with_timestamps;
    request.instructions = args.instructions.clone();
    request
}

async fn speak_long(
    composition: &compose::Composition,
    want: Option<ProviderId>,
    args: &SpeakArgs,
    default_language: &str,
    text: String,
    codec: AudioCodec,
) -> anyhow::Result<()> {
    let chunks = split_long_text(&text, args.chunk_chars.max(200));
    if chunks.is_empty() {
        bail!("no text to synthesize");
    }
    let out = args
        .out
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("speech.{}", codec.extension())));
    let temp = tempfile::tempdir()?;
    let mut chunk_paths = Vec::with_capacity(chunks.len());
    let mut manifest = Vec::with_capacity(chunks.len());
    for (index, chunk) in chunks.iter().enumerate() {
        let request = speech_request_from_args(args, default_language, chunk.clone(), codec);
        let output = composition
            .engine
            .speak(request, want)
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let path = temp
            .path()
            .join(format!("chunk-{index:04}.{}", output.codec.extension()));
        std::fs::write(&path, &output.bytes)
            .with_context(|| format!("writing {}", path.display()))?;
        manifest.push(serde_json::json!({
            "index": index,
            "chars": chunk.chars().count(),
            "bytes": output.bytes.len(),
            "provider": output.provider,
            "codec": output.codec,
            "metadata": output.metadata,
        }));
        eprintln!(
            "chunk {}/{}: {} chars, {} bytes via {}",
            index + 1,
            chunks.len(),
            chunk.chars().count(),
            output.bytes.len(),
            output.provider
        );
        chunk_paths.push(path);
    }
    composition
        .decoder
        .concat_files(&chunk_paths, &out)
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    if let Some(path) = &args.manifest {
        std::fs::write(path, serde_json::to_string_pretty(&manifest)? + "\n")
            .with_context(|| format!("writing {}", path.display()))?;
        eprintln!("wrote {} (long-form manifest)", path.display());
    }
    eprintln!("wrote {} ({} chunks)", out.display(), chunks.len());
    println!("{}", out.display());
    Ok(())
}

fn split_long_text(text: &str, max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for paragraph in text.split("\n\n") {
        for sentence in paragraph.split_inclusive(['.', '!', '?', '\n']) {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            let needs_space = !current.is_empty();
            let next_len =
                current.chars().count() + sentence.chars().count() + usize::from(needs_space);
            if next_len > max_chars && !current.is_empty() {
                chunks.push(current.trim().to_string());
                current.clear();
            }
            if !current.is_empty() {
                current.push(' ');
            }
            if sentence.chars().count() > max_chars {
                for word in sentence.split_whitespace() {
                    let next_len = current.chars().count()
                        + word.chars().count()
                        + usize::from(!current.is_empty());
                    if next_len > max_chars && !current.is_empty() {
                        chunks.push(current.trim().to_string());
                        current.clear();
                    }
                    if !current.is_empty() {
                        current.push(' ');
                    }
                    current.push_str(word);
                }
            } else {
                current.push_str(sentence);
            }
        }
    }
    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }
    chunks
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
        smart_turn: args.smart_turn,
        smart_turn_timeout_ms: args.smart_turn_timeout_ms,
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
    request.bit_rate = args.bit_rate;
    request.optimize_streaming_latency = args.optimize_streaming_latency;

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

async fn agent(config: &Config, args: AgentArgs) -> anyhow::Result<()> {
    let provider = xai_provider(config)?;
    let codec = AudioCodec::from_str(&args.codec).map_err(anyhow::Error::msg)?;
    let mut request = RealtimeAgentRequest::text(args.text);
    request.voice = args.voice;
    request.model = args.model;
    request.instructions = args.instructions;
    request.reasoning_effort = args.reasoning_effort;
    request.output_codec = codec;
    request.output_sample_rate = args.sample_rate;
    request.input_sample_rate = args.sample_rate;
    request.text_only = args.text_only;
    request.manual_turn = args.manual_turn;
    request.vad_threshold = args.vad_threshold;
    request.vad_silence_duration_ms = args.vad_silence_ms;
    request.vad_prefix_padding_ms = args.vad_prefix_padding_ms;
    request.language_hint = args.language_hint;

    let turn = provider
        .realtime_text_turn(request)
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    if let Some(path) = &args.out {
        if !turn.audio.is_empty() {
            std::fs::write(path, &turn.audio)
                .with_context(|| format!("writing {}", path.display()))?;
            eprintln!(
                "wrote {} ({} bytes, {}, {} Hz)",
                path.display(),
                turn.audio.len(),
                turn.audio_mime,
                turn.output_sample_rate
            );
        }
    } else if !turn.audio.is_empty() {
        eprintln!(
            "received {} bytes of {}; pass --out to save raw realtime audio",
            turn.audio.len(),
            turn.audio_mime
        );
    }
    if args.json {
        println!("{}", serde_json::to_string_pretty(&turn)?);
    } else {
        println!("{}", turn.text.trim());
    }
    Ok(())
}

fn providers_list(config: &Config) -> anyhow::Result<()> {
    let composition = compose::build(config);
    println!("{:<14} {:<12} CAPABILITIES", "PROVIDER", "STATUS");
    for id in ProviderId::ALL {
        let configured = composition.configured.contains(&id);
        let status = if configured {
            "ready"
        } else {
            match id {
                ProviderId::LocalCanary if !compose::local_canary_compiled() => "not compiled",
                ProviderId::LocalCanary => "no model",
                ProviderId::LocalQwen3 if !compose::local_qwen3_compiled() => "not compiled",
                ProviderId::LocalQwen3 => "no model",
                _ => "no api key",
            }
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
        ProviderId::LocalQwen3 => {
            return "tts (local, named voices, 10 languages, offline)".to_string();
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
            ProviderId::LocalQwen3 => {
                println!("local-qwen3:");
                println!("  compiled: {}", compose::local_qwen3_compiled());
                let source = match compose::local_qwen3_source(config) {
                    Some(compose::Qwen3Source::Dir(dir)) => format!("dir: {}", dir.display()),
                    Some(compose::Qwen3Source::HfCache) => "huggingface cache (shared)".to_string(),
                    None => "missing (run: openvoice models fetch qwen3-tts)".to_string(),
                };
                println!("  model: {source}");
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
                    ProviderId::LocalCanary | ProviderId::LocalQwen3 => unreachable!(),
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
        let hf_cache = model.name == "qwen3-tts" && ov_local::models::hf_cache_present(model.repo);
        let status = if installed {
            "installed"
        } else if hf_cache {
            "hf-cache"
        } else {
            "missing"
        };
        println!("{:<16} {:<12} {}", model.name, status, model.description);
    }
    Ok(())
}

fn completions(shell: CompletionShell) -> anyhow::Result<()> {
    let mut command = Cli::command();
    let name = command.get_name().to_string();
    clap_complete::generate(shell, &mut command, name, &mut std::io::stdout());
    Ok(())
}

#[derive(serde::Serialize)]
struct VoiceRow {
    provider: String,
    id: String,
    name: String,
    kind: String,
    description: String,
}

async fn voices(config: &Config, command: VoicesCommand) -> anyhow::Result<()> {
    match command {
        VoicesCommand::List(args) => voices_list(config, args).await,
        VoicesCommand::Get { id } => {
            let provider = xai_provider(config)?;
            let voice = provider.get_custom_voice(&id).await?;
            println!("{}", serde_json::to_string_pretty(&voice)?);
            Ok(())
        }
        VoicesCommand::Clone(args) => {
            if !args.file.is_file() {
                bail!("reference audio '{}' is not readable", args.file.display());
            }
            let provider = xai_provider(config)?;
            let voice = provider
                .create_custom_voice(ov_providers::CustomVoiceCreateRequest {
                    file: AudioSource::File(args.file),
                    name: args.name,
                    description: args.description,
                    gender: args.gender,
                    accent: args.accent,
                    age: args.age,
                    language: args.language,
                    use_case: args.use_case,
                    tone: args.tone,
                })
                .await?;
            println!("{}", serde_json::to_string_pretty(&voice)?);
            Ok(())
        }
        VoicesCommand::Update(args) => {
            let provider = xai_provider(config)?;
            let voice = provider
                .update_custom_voice(
                    &args.id,
                    ov_providers::CustomVoiceUpdateRequest {
                        name: args.name,
                        description: args.description,
                        gender: args.gender,
                        accent: args.accent,
                        age: args.age,
                        language: args.language,
                        use_case: args.use_case,
                        tone: args.tone,
                    },
                )
                .await?;
            println!("{}", serde_json::to_string_pretty(&voice)?);
            Ok(())
        }
        VoicesCommand::Delete { id, yes } => {
            if !yes {
                bail!("refusing to delete custom voice '{id}' without --yes");
            }
            let provider = xai_provider(config)?;
            let deleted = provider.delete_custom_voice(&id).await?;
            println!(
                "{}",
                serde_json::json!({ "voice_id": id, "deleted": deleted })
            );
            Ok(())
        }
    }
}

async fn voices_list(config: &Config, args: VoicesListArgs) -> anyhow::Result<()> {
    let mut rows = Vec::new();
    if !args.custom_only
        && matches!(
            args.provider,
            VoiceProviderFilter::All | VoiceProviderFilter::Local
        )
    {
        for (id, description) in ov_local::models::QWEN3_TTS_VOICES {
            rows.push(VoiceRow {
                provider: "local-qwen3".into(),
                id: (*id).into(),
                name: (*id).into(),
                kind: "built-in".into(),
                description: (*description).into(),
            });
        }
    }
    if !args.custom_only
        && matches!(
            args.provider,
            VoiceProviderFilter::All | VoiceProviderFilter::Xai
        )
    {
        for (id, name, description) in ov_providers::XAI_BUILT_IN_VOICES {
            rows.push(VoiceRow {
                provider: "xai".into(),
                id: (*id).into(),
                name: (*name).into(),
                kind: "built-in".into(),
                description: (*description).into(),
            });
        }
    }
    if matches!(
        args.provider,
        VoiceProviderFilter::All | VoiceProviderFilter::Xai
    ) {
        if let Some(provider) = maybe_xai_provider(config) {
            let custom = provider
                .list_custom_voices(Some(args.limit), args.page_token.as_deref())
                .await?;
            for voice in custom.voices {
                rows.push(VoiceRow {
                    provider: "xai".into(),
                    id: voice.voice_id,
                    name: voice.name.unwrap_or_default(),
                    kind: "custom".into(),
                    description: voice
                        .description
                        .or(voice.tone)
                        .or(voice.use_case)
                        .unwrap_or_default(),
                });
            }
            if let Some(token) = custom.pagination_token {
                eprintln!("next page token: {token}");
            }
        } else if args.custom_only || args.provider == VoiceProviderFilter::Xai {
            eprintln!("warn: XAI_API_KEY missing; custom voices are not listed");
        }
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    println!(
        "{:<14} {:<12} {:<10} {:<18} DESCRIPTION",
        "PROVIDER", "KIND", "ID", "NAME"
    );
    for row in rows {
        println!(
            "{:<14} {:<12} {:<10} {:<18} {}",
            row.provider, row.kind, row.id, row.name, row.description
        );
    }
    Ok(())
}

fn maybe_xai_provider(config: &Config) -> Option<ov_providers::XaiProvider> {
    config.api_key(ProviderId::Xai).map(|key| {
        ov_providers::XaiProvider::new(
            key,
            ov_providers::XaiSettings {
                base_url: config.providers.xai.base_url.clone(),
                ws_url: config.providers.xai.ws_url.clone(),
                tts_voice: config.providers.xai.tts_voice.clone(),
            },
        )
    })
}

fn xai_provider(config: &Config) -> anyhow::Result<ov_providers::XaiProvider> {
    maybe_xai_provider(config).with_context(|| {
        format!(
            "xAI API key missing (export {}=...)",
            config.providers.xai.api_key_env
        )
    })
}

async fn smoke(config: &Config, target: SmokeTarget) -> anyhow::Result<()> {
    match target {
        SmokeTarget::Local => smoke_local(config).await,
        SmokeTarget::Xai => smoke_xai(config).await,
    }
}

async fn smoke_local(config: &Config) -> anyhow::Result<()> {
    if !compose::local_canary_compiled() || !compose::local_qwen3_compiled() {
        bail!("local smoke requires a build with local STT and local TTS features");
    }
    if !compose::local_canary_installed(config) {
        bail!("local-canary model missing (run: openvoice models fetch canary-1b-v2)");
    }
    if compose::local_qwen3_source(config).is_none() {
        bail!("local-qwen3 model missing (run: openvoice models fetch qwen3-tts)");
    }
    let composition = compose::build(config);
    let mut speech = SpeechRequest::new("Hola, esta es una prueba local de open voice.");
    speech.language = Some(Language::new("es"));
    speech.codec = AudioCodec::Wav;
    let audio = composition
        .engine
        .speak(speech, Some(ProviderId::LocalQwen3))
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let temp = tempfile::NamedTempFile::new()?.into_temp_path();
    std::fs::write(&temp, &audio.bytes).with_context(|| format!("writing {}", temp.display()))?;
    let mut transcribe = TranscribeRequest::new(AudioSource::File(temp.to_path_buf()));
    transcribe.language = Some(Language::new("es"));
    let transcript = composition
        .engine
        .transcribe(transcribe, Some(ProviderId::LocalCanary))
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    println!("local smoke ok: {}", transcript.text.trim());
    Ok(())
}

async fn smoke_xai(config: &Config) -> anyhow::Result<()> {
    let _ = xai_provider(config)?;
    let composition = compose::build(config);
    let mut speech = SpeechRequest::new("Open voice xAI smoke test.");
    speech.language = Some(Language::new("en"));
    speech.codec = AudioCodec::Mp3;
    let audio = composition
        .engine
        .speak(speech, Some(ProviderId::Xai))
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let mut transcribe = TranscribeRequest::new(AudioSource::Bytes {
        data: audio.bytes,
        file_name: "openvoice-smoke.mp3".into(),
        mime: audio.mime,
    });
    transcribe.language = Some(Language::new("en"));
    let transcript = composition
        .engine
        .transcribe(transcribe, Some(ProviderId::Xai))
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    println!("xai smoke ok: {}", transcript.text.trim());
    Ok(())
}

async fn models_fetch(config: &Config, name: &str) -> anyhow::Result<()> {
    let model = ov_local::models::find(name)
        .with_context(|| format!("unknown model '{name}' (see: openvoice models list)"))?;
    let models_dir = config.models_dir();
    eprintln!("fetching {} into {}", model.name, models_dir.display());
    let mut last_progress: std::collections::HashMap<String, u64> =
        std::collections::HashMap::new();
    let dir = model
        .fetch(
            &models_dir,
            "https://huggingface.co",
            |file, bytes, total| {
                const STEP: u64 = 50 * 1024 * 1024;
                let done = total == Some(bytes);
                let last = last_progress.entry(file.to_string()).or_insert(0);
                if !done && bytes.saturating_sub(*last) < STEP {
                    return;
                }
                *last = bytes;
                match total {
                    Some(total) if total > 0 => eprintln!("  {file}: {bytes}/{total} bytes"),
                    _ => eprintln!("  {file}: {bytes} bytes"),
                }
            },
        )
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    println!("{}", dir.display());
    let (compiled, feature) = if model.name == "qwen3-tts" {
        (compose::local_qwen3_compiled(), "local-tts")
    } else {
        (compose::local_canary_compiled(), "local")
    };
    if !compiled {
        eprintln!(
            "note: this build cannot run the model (compile with --features {feature} to use it)"
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
            "mulaw",
            "--bit-rate",
            "128000",
            "--optimize-streaming-latency",
            "1",
            "--text-normalization",
            "--with-timestamps",
        ])
        .unwrap();
        match cli.command {
            Command::Speak(args) => {
                assert_eq!(args.codec, "mulaw");
                assert_eq!(args.bit_rate, Some(128_000));
                assert_eq!(args.optimize_streaming_latency, Some(1));
                assert!(args.text_normalization);
                assert!(args.with_timestamps);
            }
            _ => panic!("wrong command"),
        }

        let cli = Cli::try_parse_from([
            "openvoice",
            "stream",
            "stt",
            "a.ogg",
            "--interim",
            "--smart-turn",
            "--smart-turn-timeout-ms",
            "1500",
        ])
        .unwrap();
        match cli.command {
            Command::Stream {
                command: StreamCommand::Stt(args),
            } => {
                assert!(args.smart_turn);
                assert_eq!(args.smart_turn_timeout_ms, Some(1500));
            }
            _ => panic!("wrong command"),
        }
    }

    #[test]
    fn parses_completion_generation() {
        let cli = Cli::try_parse_from(["openvoice", "completions", "zsh"]).unwrap();
        assert!(matches!(cli.command, Command::Completions { .. }));
    }

    #[test]
    fn parses_agent_invocation() {
        let cli = Cli::try_parse_from([
            "openvoice",
            "agent",
            "hello",
            "--voice",
            "eve",
            "--text-only",
        ])
        .unwrap();
        assert!(matches!(cli.command, Command::Agent(_)));
    }

    #[test]
    fn parses_smoke_invocation() {
        let cli = Cli::try_parse_from(["openvoice", "smoke", "local"]).unwrap();
        assert!(matches!(cli.command, Command::Smoke { .. }));
    }

    #[test]
    fn parses_voice_commands() {
        let cli =
            Cli::try_parse_from(["openvoice", "voices", "list", "--provider", "xai"]).unwrap();
        assert!(matches!(cli.command, Command::Voices { .. }));

        let cli = Cli::try_parse_from([
            "openvoice",
            "voices",
            "clone",
            "--file",
            "ref.wav",
            "--name",
            "Narrator",
            "--language",
            "en",
        ])
        .unwrap();
        assert!(matches!(cli.command, Command::Voices { .. }));
    }

    #[test]
    fn long_text_splitter_prefers_sentence_boundaries() {
        let chunks = split_long_text(
            "One sentence. Two sentence. A verylongwordwithoutspaces",
            20,
        );
        assert_eq!(chunks[0], "One sentence.");
        assert_eq!(chunks[1], "Two sentence.");
        assert!(chunks.iter().all(|c| !c.trim().is_empty()));
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
