//! # ov-local
//!
//! Local (on-device) speech engines and their model management. No Python,
//! no PyTorch: inference is Rust + ONNX Runtime (feature `canary`).
//!
//! Model management (registry, download, install checks) is always compiled
//! so `openvoice models fetch` works even in builds without the inference
//! feature.

pub mod models;

#[cfg(feature = "canary")]
pub mod canary;

#[cfg(feature = "canary")]
pub use canary::CanaryLocalProvider;

#[cfg(feature = "qwen3-tts")]
pub mod qwen3_tts;

#[cfg(feature = "qwen3-tts")]
pub use qwen3_tts::Qwen3TtsLocalProvider;
