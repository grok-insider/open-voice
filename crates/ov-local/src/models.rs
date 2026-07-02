//! Local model registry + downloader. Weights come from Hugging Face at
//! runtime and are cached under the configured models dir — they are never
//! redistributed with open-voice.

use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use ov_core::{CoreError, CoreResult};

/// One file of a local model. `repo: None` means the model's default repo.
#[derive(Debug, Clone, Copy)]
pub struct ModelFile {
    pub name: &'static str,
    pub repo: Option<&'static str>,
}

const fn file(name: &'static str) -> ModelFile {
    ModelFile { name, repo: None }
}

/// A downloadable local model.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    /// Directory name under the models dir (also the CLI name).
    pub name: &'static str,
    /// Default Hugging Face repo id.
    pub repo: &'static str,
    /// Files to download into the model directory.
    pub files: &'static [ModelFile],
    pub description: &'static str,
}

/// NVIDIA Canary 1B v2, int8 ONNX export (25 European languages, the model
/// that produced the best Spanish results in our evaluation). Layout matches
/// what `transcribe-rs` expects. `nemo128.onnx` (the shared NeMo log-mel
/// preprocessor) is not present in the canary repo — it ships in the sibling
/// parakeet export.
pub const CANARY_1B_V2: ModelSpec = ModelSpec {
    name: "canary-1b-v2",
    repo: "istupakov/canary-1b-v2-onnx",
    files: &[
        file("encoder-model.int8.onnx"),
        file("decoder-model.int8.onnx"),
        ModelFile {
            name: "nemo128.onnx",
            repo: Some("istupakov/parakeet-tdt-0.6b-v3-onnx"),
        },
        file("vocab.txt"),
    ],
    description: "NVIDIA Canary 1B v2 (int8 ONNX) — multilingual STT, 25 languages",
};

pub const ALL_MODELS: &[&ModelSpec] = &[&CANARY_1B_V2];

pub fn find(name: &str) -> Option<&'static ModelSpec> {
    ALL_MODELS.iter().copied().find(|m| m.name == name)
}

impl ModelSpec {
    pub fn dir(&self, models_dir: &Path) -> PathBuf {
        models_dir.join(self.name)
    }

    /// A model is installed when every expected file exists and is non-empty.
    pub fn is_installed(&self, models_dir: &Path) -> bool {
        let dir = self.dir(models_dir);
        self.files.iter().all(|f| {
            std::fs::metadata(dir.join(f.name))
                .map(|m| m.is_file() && m.len() > 0)
                .unwrap_or(false)
        })
    }

    fn url_for(&self, base_url: &str, file: &ModelFile) -> String {
        let repo = file.repo.unwrap_or(self.repo);
        format!("{base_url}/{repo}/resolve/main/{}", file.name)
    }

    /// Download all model files into the models dir. `base_url` is
    /// `https://huggingface.co` in production and a mock server in tests.
    /// Existing complete files are skipped, so retries resume cheaply.
    pub async fn fetch(
        &self,
        models_dir: &Path,
        base_url: &str,
        mut progress: impl FnMut(&str, u64),
    ) -> CoreResult<PathBuf> {
        let dir = self.dir(models_dir);
        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(|e| CoreError::Io(format!("creating {}: {e}", dir.display())))?;

        let client = reqwest::Client::builder()
            .user_agent(concat!("open-voice/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| CoreError::Network(e.to_string()))?;

        for file in self.files {
            let target = dir.join(file.name);
            if std::fs::metadata(&target)
                .map(|m| m.is_file() && m.len() > 0)
                .unwrap_or(false)
            {
                progress(file.name, 0);
                continue;
            }
            let url = self.url_for(base_url, file);
            let response = client
                .get(&url)
                .send()
                .await
                .map_err(|e| CoreError::Network(format!("GET {url}: {e}")))?;
            if !response.status().is_success() {
                return Err(CoreError::Provider {
                    provider: "huggingface".into(),
                    message: format!("GET {url}: HTTP {}", response.status().as_u16()),
                });
            }

            // Stream to a .part file, then rename, so partial downloads never
            // masquerade as installed files.
            let partial = dir.join(format!("{}.part", file.name));
            let mut out = tokio::fs::File::create(&partial)
                .await
                .map_err(|e| CoreError::Io(format!("creating {}: {e}", partial.display())))?;
            let mut stream = response.bytes_stream();
            let mut written = 0u64;
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| CoreError::Network(format!("reading {url}: {e}")))?;
                tokio::io::AsyncWriteExt::write_all(&mut out, &chunk)
                    .await
                    .map_err(|e| CoreError::Io(format!("writing {}: {e}", partial.display())))?;
                written += chunk.len() as u64;
            }
            tokio::io::AsyncWriteExt::flush(&mut out)
                .await
                .map_err(|e| CoreError::Io(e.to_string()))?;
            drop(out);
            tokio::fs::rename(&partial, &target)
                .await
                .map_err(|e| CoreError::Io(format!("renaming {}: {e}", partial.display())))?;
            progress(file.name, written);
        }
        Ok(dir)
    }

    pub fn remove(&self, models_dir: &Path) -> CoreResult<()> {
        let dir = self.dir(models_dir);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)
                .map_err(|e| CoreError::Io(format!("removing {}: {e}", dir.display())))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_lookup() {
        assert!(find("canary-1b-v2").is_some());
        assert!(find("nope").is_none());
    }

    #[test]
    fn install_detection_requires_all_files() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!CANARY_1B_V2.is_installed(dir.path()));
        let model_dir = CANARY_1B_V2.dir(dir.path());
        std::fs::create_dir_all(&model_dir).unwrap();
        for file in CANARY_1B_V2.files {
            std::fs::write(model_dir.join(file.name), b"x").unwrap();
        }
        assert!(CANARY_1B_V2.is_installed(dir.path()));
        // An empty file is not "installed".
        std::fs::write(model_dir.join("vocab.txt"), b"").unwrap();
        assert!(!CANARY_1B_V2.is_installed(dir.path()));
    }

    #[test]
    fn url_shape() {
        let url = CANARY_1B_V2.url_for("https://huggingface.co", &file("vocab.txt"));
        assert_eq!(
            url,
            "https://huggingface.co/istupakov/canary-1b-v2-onnx/resolve/main/vocab.txt"
        );
        // Per-file repo override (the shared NeMo preprocessor).
        let preprocessor = CANARY_1B_V2
            .files
            .iter()
            .find(|f| f.name == "nemo128.onnx")
            .unwrap();
        let url = CANARY_1B_V2.url_for("https://huggingface.co", preprocessor);
        assert_eq!(
            url,
            "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/nemo128.onnx"
        );
    }
}
