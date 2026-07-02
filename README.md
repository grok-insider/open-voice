# open-voice

Agnostic speech-to-text + text-to-speech from one CLI. Local Rust/ONNX
engines and remote providers (OpenAI, ElevenLabs, Cartesia, xAI) behind one
normalized transcript/audio model — same inputs, same outputs, any engine.

```bash
# Speech-to-text: txt/srt/vtt/json next to the input
openvoice transcribe "WhatsApp Ptt 2026-07-02 at 14.56.03.ogg" --lang es --format txt,srt,json

# Text-to-speech
openvoice speak "Hola mundo" --lang es --out hola.mp3

# Realtime (xAI WebSocket)
openvoice stream stt clip.ogg --lang es --interim
openvoice stream tts "Hola mundo" --lang es --out hola.mp3

# Introspection
openvoice providers list
openvoice providers doctor
openvoice models fetch canary-1b-v2
```

## Engines

| Engine | STT | TTS | Streaming | Diarization | Word timestamps | Notes |
|---|---|---|---|---|---|---|
| `local-canary` | ✓ | — | — | — | segments | NVIDIA Canary 1B v2 (int8 ONNX), 25 languages, fully offline |
| `xai` | ✓ | ✓ | ✓ (STT+TTS) | ✓ | ✓ | OGG/Opus native, 500 MB uploads, keyterms, multichannel |
| `elevenlabs` | ✓ | ✓ | — | ✓ | ✓ | Scribe v2, 5 GB uploads, keyterms |
| `cartesia` | ✓ | ✓ | — | — | ✓ | ink-whisper STT, sonic TTS (voice id required) |
| `openai` | ✓ | ✓ | — | — | ✓ (whisper-1) | 25 MB cap; OGG transcoded automatically |

`--provider auto` (the default) tries engines in the order above and falls
back on provider-side failures. Capability mismatches (diarization, file
size, container format) are validated *before* any upload; inputs a provider
can't ingest are transcoded via ffmpeg automatically.

## Install

Prebuilt binaries are attached to
[GitHub Releases](https://github.com/grok-insider/open-voice/releases)
(static musl Linux, macOS, Windows). `ffmpeg` on PATH is required for
transcoding and streaming decode.

Nix (binaries come prebuilt from the `grok-insider` cachix cache):

```bash
nix run github:grok-insider/open-voice -- providers list
```

Home Manager:

```nix
inputs.open-voice.url = "github:grok-insider/open-voice";
# ...
imports = [ inputs.open-voice.homeManagerModules.default ];
programs.open-voice = {
  enable = true;
  aliases.enable = true; # tt-en/tt-es/tt-ru + sp-*
  settings.defaults.language = "es";
};
```

## Configuration

API keys are read from environment variables (never stored in config):

```bash
export XAI_API_KEY=...
export ELEVENLABS_API_KEY=...
export CARTESIA_API_KEY=...
export OPENAI_API_KEY=...
```

Optional `~/.config/open-voice/config.toml`:

```toml
[defaults]
stt_provider = "auto"   # or local-canary/xai/elevenlabs/cartesia/openai
tts_provider = "auto"
language = "es"
formats = "txt,srt"

[providers.cartesia]
tts_voice = "your-voice-id"

[local]
models_dir = "~/.local/share/open-voice/models"
```

## Local (offline) speech-to-text

```bash
openvoice models fetch canary-1b-v2   # ~1 GB from Hugging Face, cached locally
openvoice transcribe clip.ogg --lang es --provider local-canary
```

Local inference is compiled in with `cargo build --features local` (the
`ort` crate downloads ONNX Runtime at build time, so the Nix package and the
static release binaries exclude it; `models fetch` works in every build).

## Architecture

Hexagonal: `ov-core` defines the domain + port traits, adapters implement
them (`ov-providers`, `ov-local`, `ov-audio`), `ov-engine` holds the
use-cases (selection, validation, fallback), and the CLI is the composition
root. Adding a provider is a new adapter module + one registration line —
see [AGENTS.md](AGENTS.md).

## License

MIT.
