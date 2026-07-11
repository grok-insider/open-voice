# Changelog

All notable, user-facing changes to open-voice are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.0.1

Initial public line of the open-voice STT/TTS engine:

- Agnostic STT/TTS workspace with CLI (`openvoice`)
- xAI voice discovery, custom voices, advanced options, and realtime agent turn
- Optional local TTS (Qwen3 via any-tts/Candle) and local ONNX feature path
- Nix flake + Home-Manager module; GitHub Release binaries (musl/darwin/windows)
- Distributed via GitHub Releases and Cachix — not crates.io
