# open-voice

Agnostic speech-to-text + text-to-speech from one CLI. Local Rust/ONNX
engines and remote providers (OpenAI, ElevenLabs, Cartesia, xAI) behind one
normalized transcript/audio model — same inputs, same outputs, any engine.

```bash
# Speech-to-text: txt/srt/vtt/json next to the input
openvoice transcribe "WhatsApp Ptt 2026-07-02 at 14.56.03.ogg" --lang es --format txt,srt,json

# Text-to-speech
openvoice speak "Hola mundo" --lang es --out hola.mp3
openvoice speak --file script.txt --long --chunk-chars 1200 --manifest script.json --out script.mp3

# Voice discovery + xAI Custom Voices
openvoice voices list
openvoice voices clone --file reference.wav --name "Brand narrator" --language en

# Realtime (xAI WebSocket)
openvoice stream stt clip.ogg --lang es --interim
openvoice stream tts "Hola mundo" --lang es --out hola.mp3

# Introspection
openvoice providers list
openvoice providers doctor
openvoice models fetch canary-1b-v2
openvoice smoke local

# Shell completions
openvoice completions zsh > ~/.local/share/zsh/site-functions/_openvoice
```

## Engines

| Engine | STT | TTS | Streaming | Diarization | Word timestamps | Notes |
|---|---|---|---|---|---|---|
| `local-canary` | ✓ | — | — | — | segments | NVIDIA Canary 1B v2 (int8 ONNX), 25 languages, fully offline |
| `local-qwen3` | — | ✓ | — | — | — | Qwen3-TTS 1.7B CustomVoice (any-tts/Candle), named voices, 10 languages, offline, optional CUDA |
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
# With local (offline) Canary STT compiled in, linked against nixpkgs' ONNX Runtime:
nix run github:grok-insider/open-voice#open-voice-local -- providers list
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

Local STT is compiled in with `cargo build --features local` (the `ort`
crate downloads ONNX Runtime at build time, so the static release binaries
exclude it; `models fetch` works in every build). The Nix packages
`open-voice-local` / `open-voice-local-cuda` include it, linked against
nixpkgs' ONNX Runtime.

## Local (offline) text-to-speech

```bash
openvoice models fetch qwen3-tts      # ~3.6 GB (or reuse a warm HF cache)
openvoice speak "Hola mundo" --lang es --voice serena --provider local-qwen3 --out hola.mp3
```

Qwen3-TTS-12Hz-1.7B-CustomVoice via [any-tts](https://github.com/TM9657/any-tts)
(Candle, pure Rust). Named speakers: `ryan`, `serena`, `vivian`, `uncle_fu`,
`aiden`, `ono_anna`, `sohee`, `eric`, `dylan`; style control via
`--instructions`. Compiled in with `--features local-tts` (CPU) or
`local-tts-cuda` (GPU; `CUDA_COMPUTE_CAP` at build time — the
`open-voice-local-cuda` Nix package defaults to sm_120). Model files resolve
from an explicit `local.tts_model_dir`, the open-voice models dir, or the
shared Hugging Face cache — in that order.

## xAI Custom Voices

```bash
export XAI_API_KEY=...
openvoice voices list --provider xai
openvoice voices clone --file reference.wav --name "Support voice" --language en --tone warm
openvoice speak "Hello from our cloned voice" --provider xai --voice <voice_id> --out hello.mp3
openvoice voices delete <voice_id> --yes
```

Custom voice IDs returned by xAI are used with `--voice`, the same as built-in
voices (`eve`, `ara`, `rex`, `sal`, `leo`). xAI currently limits reference
audio to 120 seconds and custom voices to 30 per team.

xAI-specific TTS options are exposed directly:

```bash
openvoice speak "Order 123 is ready [pause] thank you." \
  --provider xai --voice <voice_id> --codec mulaw --sample-rate 8000 \
  --text-normalization --with-timestamps --out call.mulaw

openvoice stream stt call.wav --provider xai --smart-turn --smart-turn-timeout-ms 1500
```

When `--with-timestamps` returns a JSON response, open-voice writes the audio
file plus a `<audio-extension>.json` sidecar with timestamp metadata.

Long-form TTS (`--long`) splits text on paragraph/sentence boundaries,
synthesizes each chunk, then stitches the chunks with ffmpeg. `--manifest`
writes per-chunk provider/codec/metadata details. Model downloads resume from
`.part` files when Hugging Face supports range requests and print throttled
progress for large files.

Smoke checks are opt-in because they may load local models or call paid APIs:

```bash
openvoice smoke local   # local Qwen3 TTS -> local Canary STT round trip
openvoice smoke xai     # xAI TTS -> xAI STT round trip, requires XAI_API_KEY
```

## Architecture

Hexagonal: `ov-core` defines the domain + port traits, adapters implement
them (`ov-providers`, `ov-local`, `ov-audio`), `ov-engine` holds the
use-cases (selection, validation, fallback), and the CLI is the composition
root. Adding a provider is a new adapter module + one registration line —
see [AGENTS.md](AGENTS.md).

## License

MIT.
