//! Provider identity. A small closed enum (not strings) so `match` stays
//! exhaustive when a new provider is added.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderId {
    LocalCanary,
    LocalQwen3,
    Openai,
    Elevenlabs,
    Cartesia,
    Xai,
}

impl ProviderId {
    pub const ALL: [ProviderId; 6] = [
        ProviderId::LocalCanary,
        ProviderId::LocalQwen3,
        ProviderId::Openai,
        ProviderId::Elevenlabs,
        ProviderId::Cartesia,
        ProviderId::Xai,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            ProviderId::LocalCanary => "local-canary",
            ProviderId::LocalQwen3 => "local-qwen3",
            ProviderId::Openai => "openai",
            ProviderId::Elevenlabs => "elevenlabs",
            ProviderId::Cartesia => "cartesia",
            ProviderId::Xai => "xai",
        }
    }
}

impl fmt::Display for ProviderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ProviderId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "local-canary" | "local" | "canary" => Ok(ProviderId::LocalCanary),
            "local-qwen3" | "qwen3" | "qwen3-tts" => Ok(ProviderId::LocalQwen3),
            "openai" | "whisper" => Ok(ProviderId::Openai),
            "elevenlabs" | "11labs" => Ok(ProviderId::Elevenlabs),
            "cartesia" => Ok(ProviderId::Cartesia),
            "xai" | "grok" => Ok(ProviderId::Xai),
            other => Err(format!(
                "unknown provider '{other}' (expected one of: local-canary, local-qwen3, openai, elevenlabs, cartesia, xai)"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_aliases() {
        for p in ProviderId::ALL {
            assert_eq!(p.as_str().parse::<ProviderId>().unwrap(), p);
        }
        assert_eq!("grok".parse::<ProviderId>().unwrap(), ProviderId::Xai);
        assert_eq!("whisper".parse::<ProviderId>().unwrap(), ProviderId::Openai);
        assert!("nope".parse::<ProviderId>().is_err());
    }
}
