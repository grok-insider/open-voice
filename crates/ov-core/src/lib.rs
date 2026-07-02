//! # ov-core
//!
//! Domain model + port traits for open-voice. This crate owns the shared
//! vocabulary (transcripts, speech requests, capabilities, errors) and the
//! trait *ports* every adapter implements. It performs **no I/O** and depends
//! on no HTTP/ONNX/audio library — that is the whole point: adapters plug in
//! at the edges, the engine and CLI only ever see these types.

pub mod capabilities;
pub mod domain;
pub mod error;
pub mod ports;
pub mod provider;

pub use capabilities::ProviderCapabilities;
pub use error::{CoreError, CoreResult};
pub use provider::ProviderId;
