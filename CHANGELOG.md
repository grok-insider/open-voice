# Changelog

All notable, user-facing changes to open-voice are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0

- Added initial open-voice workspace with agnostic STT/TTS support for OpenAI, ElevenLabs, Cartesia, and xAI providers
- Added local ONNX-based speech engine using Canary 1B v2 via transcribe-rs (behind the `local` feature flag)
- Added `transcribe`, `speak`, `stream`, `providers`, and `models` CLI commands
- Added automatic audio transcoding and engine fallback when preferred engine is unavailable
- Added output format support for txt, srt, vtt, and json
- Added declarative Home Manager module with optional config and `tt-*`/`sp-*` aliases
- Added CI pipeline with fmt, clippy, hermetic tests, and release automation via release-plz
- Added Nix flake packaging with ffmpeg wrapped onto PATH for x86_64 and aarch64
- Fixed output filename handling to preserve interior dots (e.g., `14.56.03.ogg` → `14.56.03.txt`)
