# AGENTS.md

Instructions for AI agents and contributors working on **open-voice**.

## Project overview

open-voice is a **Rust** voice toolkit: agnostic **speech-to-text** and
**text-to-speech** behind one CLI (`openvoice`) and one normalized domain
model. Engines are pluggable:

- **Local**: NVIDIA Canary 1B v2 via ONNX Runtime (feature `local`) — no
  Python, no PyTorch, weights downloaded at runtime.
- **Remote**: OpenAI (Whisper/gpt-4o/tts), ElevenLabs (Scribe/multilingual),
  Cartesia (ink-whisper/sonic), xAI (Grok STT/TTS, batch **and** WebSocket
  streaming).

It replaced a Python (NeMo/transformers) speech stack; the golden requirement
is Spanish quality on real WhatsApp voice notes with correct output naming
(`... 14.56.03.ogg` → `... 14.56.03.txt`).

- **License: MIT.** Cargo workspace, one crate per concern.
- **Ports & adapters (hexagonal) + SOLID.** Capabilities are trait *ports* in
  `ov-core`; concrete *adapters* implement them; the engine depends only on
  ports; the CLI is the composition root. Adding a provider = a new adapter,
  not a core edit.

## Module layout

| Crate | Owns | Implements |
|-------|------|------------|
| `crates/ov-core` | Domain model (`Transcript`, `Segment`, `Word`, `SpeechRequest`...), **port traits**, `CoreError`, capabilities. **No I/O, no heavy deps.** | — (defines ports) |
| `crates/ov-audio` | ffmpeg decode/probe/transcode adapter + WAV helpers. | `AudioDecoder` |
| `crates/ov-output` | txt/srt/vtt/json writers + output-path rules. | — |
| `crates/ov-config` | Config schema, XDG paths, API-key resolution (env-first). | — |
| `crates/ov-local` | Local model registry/downloader (always) + Canary ONNX STT (feature `canary`) + Qwen3-TTS via any-tts/Candle (feature `qwen3-tts`, `qwen3-tts-cuda`). | `BatchTranscriber`, `BatchSpeechSynthesizer` |
| `crates/ov-providers` | Remote adapters: `openai`, `elevenlabs`, `cartesia`, `xai` (xAI also streams over WebSocket). | `BatchTranscriber`, `BatchSpeechSynthesizer`, `StreamingTranscriber`, `StreamingSpeechSynthesizer` |
| `crates/ov-engine` | Use-cases: provider selection, capability validation, auto-transcode, `auto` fallback chains. **Depends only on `ov-core`.** | — (consumes ports) |
| `crates/ov-cli` | The `openvoice` binary: subcommands + the composition root (`compose.rs`). Features: `local` (Canary STT), `local-tts` (Qwen3 TTS CPU), `local-tts-cuda`. | — |

### The dependency rule (do not break this)

```
ov-cli ──▶ ov-engine ──▶ ov-core ◀── every adapter crate
   │ (composition root names adapters)
   └──▶ ov-audio / ov-output / ov-config / ov-local / ov-providers
```

- `ov-core` depends on nothing internal.
- `ov-engine` depends on **only** `ov-core`. Never `use` an adapter crate from
  it — add a new port in `ov-core` instead.
- Adapter crates depend on `ov-core` (+ their own I/O deps), never on each
  other or on `ov-engine`.
- `ov-cli::compose` is the only place that names concrete adapters, in `auto`
  preference order (STT: local-canary → xai → elevenlabs → cartesia → openai;
  TTS: local-qwen3 → xai → elevenlabs → cartesia → openai).

## Coding standards

- **Errors:** ports return `ov_core::CoreResult<T>`. Adapters convert
  explicitly at the boundary (HTTP 401/403 → `Auth`, 429 → `RateLimited`,
  capability mismatch → `Unsupported`...). `CoreError::is_fallback_worthy`
  drives the `auto` chain — keep the mapping honest.
- **Secrets:** API keys come from env vars via `ov-config`; never log a key,
  never put keys in the Nix store or in fixtures.
- **Capability honesty:** each adapter's `capabilities()` must reflect the
  provider's real documented limits (upload size, extensions, diarization...)
  — the engine pre-validates against them for actionable errors.
- **Tests are hermetic:** wiremock for HTTP adapters, an in-process
  tokio-tungstenite server for xAI streaming, ffmpeg tests skip when ffmpeg is
  absent. No live-provider calls in `cargo test`.
- **Formatting/lints:** `cargo fmt` + `cargo clippy --workspace --all-targets
  -- -D warnings` must be clean.
- Comments explain *why* (a provider quirk, a protocol constraint), not *what*.

## Provider quirks worth knowing

- **xAI STT multipart:** the `file` field must be the **last** multipart field
  (tested in `tests/xai_batch.rs` — keep it that way).
- **OpenAI:** only `whisper-1` yields verbose_json + word/segment timestamps;
  OGG isn't in its accepted-extension list, so the engine transcodes OGG to
  WAV first (25 MB cap validated before upload).
- **ElevenLabs:** STT word list contains `word`/`spacing`/`audio_event`
  entries — only `word` maps to domain words. `eleven_multilingual_v2` rejects
  `language_code`.
- **Cartesia:** every call needs the `Cartesia-Version` header; TTS requires
  an explicit voice id (fail fast with `NotConfigured`).
- **Local Canary:** consumes 16 kHz mono WAV (ffmpeg-decoded), int8 ONNX from
  `istupakov/canary-1b-v2-onnx`; the `ort` build script downloads ONNX Runtime
  at build time (hence feature-gated off default builds; devshell provides
  pkg-config + openssl for it).
- **Local Qwen3-TTS:** `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` via any-tts
  (Candle). Speaker ids are lowercase (`ryan`, not `Ryan`); language tags map
  ISO → names (`es` → `Spanish`); the model natively emits 24 kHz WAV and
  other codecs go through the `AudioEncoder` port (ffmpeg). The 0.6B
  checkpoint does NOT load (any-tts shape mismatch) — stay on 1.7B. Model
  resolution: explicit dir → open-voice models dir → shared HF cache. CUDA:
  build with `CUDA_COMPUTE_CAP` + toolkit stubs for `-lcuda`, run with
  `/run/opengl-driver/lib` on LD_LIBRARY_PATH (flake handles both); the CUDA
  package is not built in CI — push it to cachix from a dev machine.

## Commands

```bash
cargo build                                            # debug build (no local ONNX)
cargo test --workspace                                 # hermetic suite
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
cargo check -p ov-cli --features local                 # local Canary engine (needs pkg-config+openssl)
nix build .#open-voice -L                              # packaged build (matches CI)
nix flake check --no-build                             # validate flake outputs

# Smoke tests (need keys / model):
openvoice providers doctor
openvoice models fetch canary-1b-v2
openvoice transcribe clip.ogg --lang es --format txt,srt,json
openvoice speak "Hola mundo" --lang es --provider xai --out hola.mp3
openvoice stream stt clip.ogg --lang es --interim
```

## Releases, CI & branch protection

- **`master` is protected:** changes land via **PR**, `fmt + clippy + test`
  must pass. See the org-wide `~/dev/opensource/AGENTS.md` for the full git +
  release rules (Conventional Commits, signed, squash-merge...).
- **CI** (`.github/workflows/ci.yml`): fmt + clippy + hermetic tests on every
  PR; a non-required job checks `--features local` still compiles; on
  master/tags it `nix build`s (x86_64 + aarch64) and pushes to the
  `grok-insider` cachix cache.
- **Releases** (`.github/workflows/release.yml` + `release-plz.toml`),
  open-recorder's model: a **hand-rolled** `release-pr` job keeps a patch-line
  release PR updated (version bump + Cargo.lock + AI changelog) — NOT
  `release-plz release-pr`, whose `cargo package` change detection can't
  resolve our unpublished internal path deps against crates.io. Deliberate
  minor/major bumps go through `manual-version-bump.yml`. Merging a release PR
  lands an untagged version; `release-plz release` (`release_always = true`)
  then tags `vX.Y.Z`, creates the GitHub Release, and a cross-platform matrix
  attaches static-musl Linux, macOS, and Windows `openvoice` binaries (default
  features — no ONNX in release artifacts).
- Two release gotchas learned cutting v0.1.0 (do not regress): `ov-cli`'s
  Cargo manifest must NOT set `publish = false` (release-plz would skip its
  git release — "nothing to release"), and the crates.io guard lives in
  `release-plz.toml` instead.
- Required repo secrets: `RELEASE_PLZ_TOKEN`, `OPENROUTER_API_KEY`,
  `CACHIX_AUTH_TOKEN`.

## Conventions

- Repo: `github.com/grok-insider/open-voice`. Binary: `openvoice`.
- Crate prefix `ov-`. Provider modules are lowercase provider names.
- Prefer fixing the contract in `ov-core` over working around it in an
  adapter.
- Model weights are never committed or redistributed; they download from
  Hugging Face at runtime into the XDG data dir.
