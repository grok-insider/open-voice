//! # ov-providers
//!
//! Remote provider adapters. One module per provider; each maps its wire
//! format into `ov_core` domain types and its HTTP/WS failures into
//! `CoreError` variants at the boundary. Nothing outside this crate knows any
//! provider-specific JSON shape (OCP: adding a provider = adding a module).

pub mod cartesia;
pub mod elevenlabs;
mod http;
pub mod openai;
pub mod xai;

pub use cartesia::{CartesiaProvider, CartesiaSettings};
pub use elevenlabs::{ElevenLabsProvider, ElevenLabsSettings};
pub use openai::{OpenAiProvider, OpenAiSettings};
pub use xai::{XaiProvider, XaiSettings};
