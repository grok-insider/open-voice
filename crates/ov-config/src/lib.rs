//! # ov-config
//!
//! Configuration schema + loading for open-voice.
//!
//! Secrets policy: API keys are resolved from **environment variables** by
//! default (`api_key_env`). A literal `api_key` field exists for
//! containers/CI but is discouraged; keys never belong in the Nix store.

use std::path::{Path, PathBuf};

use ov_core::{CoreError, CoreResult, ProviderId};
use serde::{Deserialize, Serialize};

pub const CONFIG_ENV: &str = "OPENVOICE_CONFIG";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub defaults: Defaults,
    pub providers: Providers,
    pub local: LocalConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Defaults {
    /// `auto` or a concrete provider id.
    pub stt_provider: String,
    pub tts_provider: String,
    /// Default language hint (ISO 639-1), empty = none.
    pub language: String,
    /// Default output formats for `transcribe`.
    pub formats: String,
}

impl Default for Defaults {
    fn default() -> Self {
        Defaults {
            stt_provider: "auto".to_string(),
            tts_provider: "auto".to_string(),
            language: String::new(),
            formats: "txt,srt".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct Providers {
    pub openai: OpenAiConfig,
    pub elevenlabs: ElevenLabsConfig,
    pub cartesia: CartesiaConfig,
    pub xai: XaiConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OpenAiConfig {
    pub api_key_env: String,
    pub api_key: Option<String>,
    pub base_url: String,
    pub stt_model: String,
    pub tts_model: String,
    pub tts_voice: String,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        OpenAiConfig {
            api_key_env: "OPENAI_API_KEY".to_string(),
            api_key: None,
            base_url: "https://api.openai.com".to_string(),
            // whisper-1 is the only model with verbose_json + word/segment
            // timestamps, which srt/vtt need.
            stt_model: "whisper-1".to_string(),
            tts_model: "gpt-4o-mini-tts".to_string(),
            tts_voice: "marin".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ElevenLabsConfig {
    pub api_key_env: String,
    pub api_key: Option<String>,
    pub base_url: String,
    pub stt_model: String,
    pub tts_model: String,
    /// Default ElevenLabs voice id ("Rachel", the ElevenLabs default voice).
    pub tts_voice: String,
}

impl Default for ElevenLabsConfig {
    fn default() -> Self {
        ElevenLabsConfig {
            api_key_env: "ELEVENLABS_API_KEY".to_string(),
            api_key: None,
            base_url: "https://api.elevenlabs.io".to_string(),
            stt_model: "scribe_v2".to_string(),
            tts_model: "eleven_multilingual_v2".to_string(),
            tts_voice: "21m00Tcm4TlvDq8ikWAM".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CartesiaConfig {
    pub api_key_env: String,
    pub api_key: Option<String>,
    pub base_url: String,
    /// `Cartesia-Version` header value.
    pub version: String,
    pub stt_model: String,
    pub tts_model: String,
    /// Cartesia requires an explicit voice id; empty means unset.
    pub tts_voice: String,
}

impl Default for CartesiaConfig {
    fn default() -> Self {
        CartesiaConfig {
            api_key_env: "CARTESIA_API_KEY".to_string(),
            api_key: None,
            base_url: "https://api.cartesia.ai".to_string(),
            version: "2026-03-01".to_string(),
            stt_model: "ink-whisper".to_string(),
            tts_model: "sonic-3.5".to_string(),
            tts_voice: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct XaiConfig {
    pub api_key_env: String,
    pub api_key: Option<String>,
    pub base_url: String,
    pub ws_url: String,
    pub tts_voice: String,
}

impl Default for XaiConfig {
    fn default() -> Self {
        XaiConfig {
            api_key_env: "XAI_API_KEY".to_string(),
            api_key: None,
            base_url: "https://api.x.ai".to_string(),
            ws_url: "wss://api.x.ai".to_string(),
            tts_voice: "eve".to_string(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LocalConfig {
    /// Directory holding local model directories (canary-1b-v2, ...).
    pub models_dir: String,
}

impl Config {
    /// Default config file path: `$XDG_CONFIG_HOME/open-voice/config.toml`.
    pub fn default_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("open-voice").join("config.toml"))
    }

    /// Load from `OPENVOICE_CONFIG`, the default path, or fall back to
    /// defaults when no file exists.
    pub fn load() -> CoreResult<Config> {
        if let Ok(path) = std::env::var(CONFIG_ENV) {
            return Config::load_from(Path::new(&path));
        }
        match Config::default_path() {
            Some(path) if path.exists() => Config::load_from(&path),
            _ => Ok(Config::default()),
        }
    }

    pub fn load_from(path: &Path) -> CoreResult<Config> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| CoreError::Io(format!("reading {}: {e}", path.display())))?;
        toml::from_str(&raw)
            .map_err(|e| CoreError::InvalidInput(format!("parsing {}: {e}", path.display())))
    }

    /// Resolve the models dir, defaulting to
    /// `$XDG_DATA_HOME/open-voice/models`.
    pub fn models_dir(&self) -> PathBuf {
        if !self.local.models_dir.is_empty() {
            return expand_tilde(&self.local.models_dir);
        }
        dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("open-voice")
            .join("models")
    }

    /// Resolve the API key for `provider`: literal `api_key` first, then the
    /// configured environment variable.
    pub fn api_key(&self, provider: ProviderId) -> Option<String> {
        let (literal, env_name) = match provider {
            ProviderId::Openai => (
                self.providers.openai.api_key.clone(),
                self.providers.openai.api_key_env.clone(),
            ),
            ProviderId::Elevenlabs => (
                self.providers.elevenlabs.api_key.clone(),
                self.providers.elevenlabs.api_key_env.clone(),
            ),
            ProviderId::Cartesia => (
                self.providers.cartesia.api_key.clone(),
                self.providers.cartesia.api_key_env.clone(),
            ),
            ProviderId::Xai => (
                self.providers.xai.api_key.clone(),
                self.providers.xai.api_key_env.clone(),
            ),
            ProviderId::LocalCanary => return None,
        };
        literal
            .filter(|k| !k.is_empty())
            .or_else(|| std::env::var(env_name).ok().filter(|k| !k.is_empty()))
    }
}

fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn defaults_are_sane() {
        let c = Config::default();
        assert_eq!(c.defaults.stt_provider, "auto");
        assert_eq!(c.providers.xai.base_url, "https://api.x.ai");
        assert_eq!(c.providers.cartesia.version, "2026-03-01");
        assert_eq!(c.providers.openai.stt_model, "whisper-1");
    }

    #[test]
    fn loads_partial_toml() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(
            f,
            r#"
[defaults]
stt_provider = "xai"

[providers.xai]
api_key = "test-key"

[local]
models_dir = "/models"
"#
        )
        .unwrap();
        let c = Config::load_from(f.path()).unwrap();
        assert_eq!(c.defaults.stt_provider, "xai");
        // Untouched sections keep their defaults.
        assert_eq!(c.defaults.formats, "txt,srt");
        assert_eq!(c.providers.openai.base_url, "https://api.openai.com");
        assert_eq!(c.api_key(ProviderId::Xai).as_deref(), Some("test-key"));
        assert_eq!(c.models_dir(), PathBuf::from("/models"));
    }

    #[test]
    fn rejects_unknown_fields() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        write!(f, "[defaults]\nnot_a_field = 1\n").unwrap();
        assert!(Config::load_from(f.path()).is_err());
    }

    #[test]
    fn api_key_prefers_literal_over_env() {
        let mut c = Config::default();
        c.providers.openai.api_key = Some("literal".into());
        assert_eq!(c.api_key(ProviderId::Openai).as_deref(), Some("literal"));
        assert_eq!(c.api_key(ProviderId::LocalCanary), None);
    }
}
