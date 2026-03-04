// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-Layer Transcoder (CLT) support.
//!
//! Loads pre-trained CLT weights from `HuggingFace` (circuit-tracer format),
//! encodes residual stream activations into sparse feature activations,
//! and injects decoder vectors into the residual stream for steering.
//!
//! Memory-efficient: uses stream-and-free for encoders (~75 MB/layer on GPU)
//! and a micro-cache for steering vectors (~450 KB for 50 features).
//!
//! # CLT Architecture
//!
//! A cross-layer transcoder at layer `l` implements:
//! ```text
//! Encode:  features = ReLU(W_enc[l] @ residual_mid[l] + b_enc[l])
//! Decode:  For each downstream layer l' >= l:
//!            mlp_out_hat[l'] += W_dec[l, l'] @ features + b_dec[l']
//! Inject:  residual[pos] += strength × W_dec[l, target_layer, feature_idx, :]
//! ```
//!
//! # Weight File Layout (circuit-tracer format)
//!
//! Each encoder file `W_enc_{l}.safetensors` contains:
//! - `W_enc_{l}`: shape `[n_features, d_model]` (BF16) — encoder weight matrix
//! - `b_enc_{l}`: shape `[n_features]` (BF16) — encoder bias
//! - `b_dec_{l}`: shape `[d_model]` (BF16) — decoder bias for target layer l
//!
//! Each decoder file `W_dec_{l}.safetensors` contains:
//! - `W_dec_{l}`: shape `[n_features, n_target_layers, d_model]` (BF16)
//!   where `n_target_layers = n_layers - l` (layer l writes to layers l..n_layers-1)

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use safetensors::tensor::SafeTensors;
use tracing::info;

use crate::error::{MIError, Result};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Identifies a single CLT feature by its source layer and index within that layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CltFeatureId {
    /// Source layer where this feature's encoder lives (`0..n_layers`).
    pub layer: usize,
    /// Feature index within the layer (`0..n_features_per_layer`).
    pub index: usize,
}

impl std::fmt::Display for CltFeatureId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "L{}:{}", self.layer, self.index)
    }
}

/// Sparse representation of CLT feature activations.
///
/// Only features with non-zero activation (after `ReLU`) are stored,
/// sorted by activation magnitude in descending order.
pub struct SparseActivations {
    /// Active features with their activation magnitudes, sorted descending.
    pub features: Vec<(CltFeatureId, f32)>,
}

impl SparseActivations {
    /// Number of active features.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.features.len()
    }

    /// Whether no features are active.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Truncate to the top-k most active features.
    pub fn truncate(&mut self, k: usize) {
        self.features.truncate(k);
    }
}

/// CLT configuration auto-detected from tensor shapes.
#[derive(Debug, Clone)]
pub struct CltConfig {
    /// Number of layers in the base model (26 for Gemma 2 2B).
    pub n_layers: usize,
    /// Hidden dimension of the base model (2304 for Gemma 2 2B).
    pub d_model: usize,
    /// Number of features per encoder layer (16384 for CLT-426K).
    pub n_features_per_layer: usize,
    /// Total feature count across all layers.
    pub n_features_total: usize,
    /// Base model name from config.yaml.
    pub model_name: String,
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// Currently loaded encoder weights on GPU.
struct LoadedEncoder {
    /// Layer index this encoder corresponds to.
    layer: usize,
    /// Encoder weight matrix.
    ///
    /// # Shapes
    /// - `w_enc`: `[n_features, d_model]`
    w_enc: Tensor,
    /// Encoder bias vector.
    ///
    /// # Shapes
    /// - `b_enc`: `[n_features]`
    b_enc: Tensor,
}

// ---------------------------------------------------------------------------
// CrossLayerTranscoder
// ---------------------------------------------------------------------------

/// Cross-Layer Transcoder.
///
/// Loads CLT encoder/decoder weights on-demand from `HuggingFace` safetensors,
/// with memory-efficient streaming (only one encoder on GPU at a time)
/// and a micro-cache for steering vectors.
///
/// Downloads are lazy: [`open()`](Self::open) only fetches config and the first
/// encoder for dimension detection. Subsequent files are downloaded as needed by
/// [`load_encoder()`](Self::load_encoder).
///
/// # Example
///
/// ```no_run
/// # fn main() -> candle_mi::Result<()> {
/// use candle_mi::clt::CrossLayerTranscoder;
/// use candle_core::Device;
///
/// let mut clt = CrossLayerTranscoder::open("mntss/clt-gemma-2-2b-426k")?;
/// println!("CLT: {} layers, d_model={}", clt.config().n_layers, clt.config().d_model);
///
/// // Load encoder for layer 10
/// let device = Device::Cpu;
/// clt.load_encoder(10, &device)?;
/// # Ok(())
/// # }
/// ```
pub struct CrossLayerTranscoder {
    /// `HuggingFace` repository ID for on-demand downloads.
    repo_id: String,
    /// Fetch configuration for `hf-fetch-model` downloads.
    fetch_config: hf_fetch_model::FetchConfig,
    /// Local paths to already-downloaded encoder files (None = not yet downloaded).
    encoder_paths: Vec<Option<PathBuf>>,
    /// Auto-detected configuration.
    config: CltConfig,
    /// Currently loaded encoder (stream-and-free: only one at a time).
    loaded_encoder: Option<LoadedEncoder>,
}

impl CrossLayerTranscoder {
    /// Open a CLT from `HuggingFace` and detect its configuration.
    ///
    /// Only downloads `config.yaml` and `W_enc_0.safetensors` (~75 MB).
    /// All other encoder/decoder files are downloaded lazily on first use.
    ///
    /// # Arguments
    /// * `clt_repo` — `HuggingFace` repository ID (e.g., `"mntss/clt-gemma-2-2b-426k"`)
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Download`] if the repository is inaccessible or files
    /// cannot be fetched. Returns [`MIError::Config`] if the weight format is
    /// unexpected.
    pub fn open(clt_repo: &str) -> Result<Self> {
        let fetch_config = hf_fetch_model::FetchConfig::builder()
            .on_progress(|event| {
                tracing::info!(
                    filename = %event.filename,
                    percent = event.percent,
                    bytes_downloaded = event.bytes_downloaded,
                    bytes_total = event.bytes_total,
                    "CLT download progress",
                );
            })
            .build()
            .map_err(|e| MIError::Download(format!("failed to build fetch config: {e}")))?;

        // Detect n_layers by listing repo files (no downloads needed).
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| MIError::Download(format!("failed to create tokio runtime: {e}")))?;
        let repo_files = rt
            .block_on(hf_fetch_model::repo::list_repo_files_with_metadata(
                clt_repo, None, None,
            ))
            .map_err(|e| MIError::Download(format!("failed to list repo files: {e}")))?;
        let n_layers = repo_files
            .iter()
            .filter(|f| f.filename.starts_with("W_enc_") && f.filename.ends_with(".safetensors"))
            .count();
        if n_layers == 0 {
            return Err(MIError::Config(format!(
                "no CLT encoder files found in {clt_repo}"
            )));
        }

        // Parse config.yaml for model_name (simple line-by-line, no serde_yaml dep).
        let model_name = match hf_fetch_model::download_file_blocking(
            clt_repo.to_owned(),
            "config.yaml",
            &fetch_config,
        ) {
            Ok(path) => {
                let text = std::fs::read_to_string(&path)?;
                parse_yaml_value(&text, "model_name").unwrap_or_else(|| "unknown".to_owned())
            }
            Err(_) => "unknown".to_owned(),
        };

        // Download W_enc_0 for dimension detection (~75 MB).
        let enc0_path = hf_fetch_model::download_file_blocking(
            clt_repo.to_owned(),
            "W_enc_0.safetensors",
            &fetch_config,
        )
        .map_err(|e| MIError::Download(format!("failed to download W_enc_0: {e}")))?;

        let data = std::fs::read(&enc0_path)?;
        let tensors = SafeTensors::deserialize(&data)
            .map_err(|e| MIError::Config(format!("failed to deserialize W_enc_0: {e}")))?;
        let w_enc_view = tensors
            .tensor("W_enc_0")
            .map_err(|e| MIError::Config(format!("tensor 'W_enc_0' not found: {e}")))?;
        let shape = w_enc_view.shape();
        if shape.len() != 2 {
            return Err(MIError::Config(format!(
                "expected 2D encoder weight, got shape {shape:?}"
            )));
        }
        let n_features_per_layer = *shape
            .first()
            .ok_or_else(|| MIError::Config("encoder weight shape is empty".into()))?;
        let d_model = *shape.get(1).ok_or_else(|| {
            MIError::Config("encoder weight shape has fewer than 2 dimensions".into())
        })?;

        // Initialise paths: only first encoder known, rest downloaded lazily.
        let mut encoder_paths: Vec<Option<PathBuf>> = vec![None; n_layers];
        if let Some(slot) = encoder_paths.first_mut() {
            *slot = Some(enc0_path);
        }

        let config = CltConfig {
            n_layers,
            d_model,
            n_features_per_layer,
            n_features_total: n_layers * n_features_per_layer,
            model_name,
        };
        info!(
            "CLT config: {} layers, d_model={}, features_per_layer={}, total={}",
            config.n_layers, config.d_model, config.n_features_per_layer, config.n_features_total
        );

        Ok(Self {
            repo_id: clt_repo.to_owned(),
            fetch_config,
            encoder_paths,
            config,
            loaded_encoder: None,
        })
    }

    /// Access the auto-detected CLT configuration.
    #[must_use]
    pub const fn config(&self) -> &CltConfig {
        &self.config
    }

    /// Check whether an encoder is currently loaded and for which layer.
    #[must_use]
    pub fn loaded_encoder_layer(&self) -> Option<usize> {
        self.loaded_encoder.as_ref().map(|e| e.layer)
    }

    // --- Lazy download helpers ---

    /// Ensure the encoder file for a given layer is downloaded. Returns the path.
    fn ensure_encoder_path(&mut self, layer: usize) -> Result<PathBuf> {
        if let Some(path) = self
            .encoder_paths
            .get(layer)
            .and_then(std::option::Option::as_ref)
        {
            // BORROW: explicit .clone() — PathBuf from Vec
            return Ok(path.clone());
        }
        let filename = format!("W_enc_{layer}.safetensors");
        info!("Downloading {filename} from {}", self.repo_id);
        let path = hf_fetch_model::download_file_blocking(
            self.repo_id.clone(),
            &filename,
            &self.fetch_config,
        )
        .map_err(|e| MIError::Download(format!("failed to download {filename}: {e}")))?;
        if let Some(slot) = self.encoder_paths.get_mut(layer) {
            // BORROW: explicit .clone() — store PathBuf in cache
            *slot = Some(path.clone());
        }
        Ok(path)
    }

    // --- Encoder loading (stream-and-free) ---

    /// Load a single encoder's weights to the specified device.
    ///
    /// Frees any previously loaded encoder first (stream-and-free pattern).
    /// Peak GPU overhead: ~75 MB for CLT-426K, ~450 MB for CLT-2.5M.
    ///
    /// # Arguments
    /// * `layer` — Layer index (`0..n_layers`)
    /// * `device` — Target device (CPU or CUDA)
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Config`] if the layer is out of range.
    /// Returns [`MIError::Download`] if the encoder file cannot be fetched.
    /// Returns [`MIError::Model`] on tensor deserialization failure.
    pub fn load_encoder(&mut self, layer: usize, device: &Device) -> Result<()> {
        if layer >= self.config.n_layers {
            return Err(MIError::Config(format!(
                "layer {layer} out of range (CLT has {} layers)",
                self.config.n_layers
            )));
        }

        // Skip if already loaded.
        if let Some(ref enc) = self.loaded_encoder {
            if enc.layer == layer {
                return Ok(());
            }
        }

        // Drop previous encoder (frees GPU memory).
        self.loaded_encoder = None;

        info!("Loading CLT encoder for layer {layer}");

        let enc_path = self.ensure_encoder_path(layer)?;
        let data = std::fs::read(&enc_path)?;
        let st = SafeTensors::deserialize(&data).map_err(|e| {
            MIError::Config(format!("failed to deserialize encoder layer {layer}: {e}"))
        })?;

        let w_enc_name = format!("W_enc_{layer}");
        let b_enc_name = format!("b_enc_{layer}");

        let w_enc = tensor_from_view(
            &st.tensor(&w_enc_name)
                .map_err(|e| MIError::Config(format!("tensor '{w_enc_name}' not found: {e}")))?,
            device,
        )?;
        let b_enc = tensor_from_view(
            &st.tensor(&b_enc_name)
                .map_err(|e| MIError::Config(format!("tensor '{b_enc_name}' not found: {e}")))?,
            device,
        )?;

        self.loaded_encoder = Some(LoadedEncoder {
            layer,
            w_enc,
            b_enc,
        });

        Ok(())
    }

    // --- Encoding ---

    /// Encode a residual stream activation into sparse CLT features.
    ///
    /// The residual should be the "residual mid" activation at the given layer
    /// (after attention, before MLP).
    ///
    /// Returns all features that pass the `ReLU` threshold, sorted by
    /// activation magnitude in descending order.
    ///
    /// # Shapes
    /// - `residual`: `[d_model]` — residual stream activation at one position
    /// - returns: [`SparseActivations`] with `(CltFeatureId, f32)` pairs
    ///
    /// # Requires
    /// [`load_encoder(layer)`](Self::load_encoder) must have been called first.
    ///
    /// # Errors
    ///
    /// Returns [`MIError::Hook`] if no encoder is loaded or the wrong layer is loaded.
    /// Returns [`MIError::Model`] on tensor operation failure.
    pub fn encode(&self, residual: &Tensor, layer: usize) -> Result<SparseActivations> {
        let enc = self.loaded_encoder.as_ref().ok_or_else(|| {
            MIError::Hook(format!(
                "no encoder loaded — call load_encoder({layer}) first"
            ))
        })?;
        if enc.layer != layer {
            return Err(MIError::Hook(format!(
                "loaded encoder is for layer {}, but layer {layer} was requested",
                enc.layer
            )));
        }

        // Compute pre-activations in F32 for numerical stability.
        // W_enc: [n_features, d_model], residual: [d_model]
        // pre_acts = W_enc @ residual + b_enc → [n_features]
        let residual_f32 = residual.flatten_all()?;
        // PROMOTE: matmul and bias add require F32 for numerical stability
        let residual_f32 = residual_f32.to_dtype(DType::F32)?;
        let w_enc_f32 = enc.w_enc.to_dtype(DType::F32)?;
        let b_enc_f32 = enc.b_enc.to_dtype(DType::F32)?;

        let pre_acts = w_enc_f32.matmul(&residual_f32.unsqueeze(1)?)?.squeeze(1)?;
        let pre_acts = (&pre_acts + &b_enc_f32)?;

        // ReLU activation.
        let acts = pre_acts.relu()?;

        // Transfer to CPU for sparse extraction.
        let acts_vec: Vec<f32> = acts.to_vec1()?;

        let mut features: Vec<(CltFeatureId, f32)> = acts_vec
            .iter()
            .enumerate()
            .filter(|&(_, v)| *v > 0.0)
            .map(|(i, v)| (CltFeatureId { layer, index: i }, *v))
            .collect();

        // Sort by activation magnitude (descending).
        features.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(SparseActivations { features })
    }

    /// Encode and return only the top-k most active features.
    ///
    /// # Shapes
    /// - `residual`: `[d_model]` — residual stream activation at one position
    /// - returns: [`SparseActivations`] truncated to at most `k` entries
    ///
    /// # Requires
    /// [`load_encoder(layer)`](Self::load_encoder) must have been called first.
    ///
    /// # Errors
    ///
    /// Same as [`encode()`](Self::encode).
    pub fn top_k(&self, residual: &Tensor, layer: usize, k: usize) -> Result<SparseActivations> {
        let mut sparse = self.encode(residual, layer)?;
        sparse.truncate(k);
        Ok(sparse)
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Convert a safetensors `TensorView` to a candle `Tensor`.
///
/// # Shapes
/// - Preserves the original tensor shape from safetensors.
///
/// # Errors
///
/// Returns [`MIError::Config`] if the tensor dtype is not supported (BF16, F16, F32).
/// Returns [`MIError::Model`] on tensor construction failure.
fn tensor_from_view(view: &safetensors::tensor::TensorView<'_>, device: &Device) -> Result<Tensor> {
    let shape: Vec<usize> = view.shape().to_vec();
    #[allow(clippy::wildcard_enum_match_arm)]
    // EXHAUSTIVE: safetensors exposes many dtypes; CLTs only use float types
    let dtype = match view.dtype() {
        safetensors::Dtype::BF16 => DType::BF16,
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::F32 => DType::F32,
        other => {
            return Err(MIError::Config(format!(
                "unsupported CLT tensor dtype: {other:?}"
            )));
        }
    };
    let tensor = Tensor::from_raw_buffer(view.data(), dtype, &shape, device)?;
    Ok(tensor)
}

/// Parse a value from a simple YAML file by key.
///
/// No `serde_yaml` dependency — uses line-by-line matching.
fn parse_yaml_value(yaml_text: &str, key: &str) -> Option<String> {
    for line in yaml_text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(key) {
            if let Some(rest) = rest.strip_prefix(':') {
                let value = rest.trim().trim_matches('"');
                return Some(value.to_owned());
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn clt_feature_id_display() {
        let fid = CltFeatureId {
            layer: 5,
            index: 42,
        };
        assert_eq!(fid.to_string(), "L5:42");
    }

    #[test]
    fn clt_feature_id_ordering() {
        let a = CltFeatureId {
            layer: 0,
            index: 10,
        };
        let b = CltFeatureId {
            layer: 0,
            index: 20,
        };
        let c = CltFeatureId { layer: 1, index: 0 };
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn sparse_activations_basics() {
        let features = vec![
            (CltFeatureId { layer: 0, index: 5 }, 3.0),
            (CltFeatureId { layer: 0, index: 2 }, 2.0),
            (CltFeatureId { layer: 0, index: 8 }, 1.0),
        ];
        let sparse = SparseActivations { features };
        assert_eq!(sparse.len(), 3);
        assert!(!sparse.is_empty());
    }

    #[test]
    fn sparse_activations_truncate() {
        let features = vec![
            (CltFeatureId { layer: 0, index: 5 }, 3.0),
            (CltFeatureId { layer: 0, index: 2 }, 2.0),
            (CltFeatureId { layer: 0, index: 8 }, 1.0),
        ];
        let mut sparse = SparseActivations { features };
        sparse.truncate(2);
        assert_eq!(sparse.len(), 2);
        assert_eq!(sparse.features[0].0.index, 5);
        assert_eq!(sparse.features[1].0.index, 2);
    }

    #[test]
    fn encode_synthetic() {
        // Create a small synthetic encoder: 4 features, d_model=8
        let device = Device::Cpu;
        let d_model = 8;
        let n_features = 4;

        // W_enc: [4, 8] — identity-like rows so we can predict output
        #[rustfmt::skip]
        let w_enc_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // feature 0: picks up residual[0]
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // feature 1: picks up residual[1]
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // feature 2: picks up residual[2]
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // feature 3: picks up residual[3]
        ];
        let w_enc = Tensor::from_vec(w_enc_data, (n_features, d_model), &device).unwrap();

        // b_enc: [4] — bias shifts to test ReLU
        let b_enc_data: Vec<f32> = vec![0.0, -0.5, 0.0, -2.0]; // feature 3 will need residual[3] > 2.0
        let b_enc = Tensor::from_vec(b_enc_data, (n_features,), &device).unwrap();

        // Residual: [8] — values: [1.5, 0.3, 0.0, 1.0, ...]
        let residual_data: Vec<f32> = vec![1.5, 0.3, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let residual = Tensor::from_vec(residual_data, (d_model,), &device).unwrap();

        // Expected pre_acts = W_enc @ residual + b_enc
        // = [1.5, 0.3, 0.0, 1.0] + [0.0, -0.5, 0.0, -2.0]
        // = [1.5, -0.2, 0.0, -1.0]
        // After ReLU: [1.5, 0.0, 0.0, 0.0]
        // Only feature 0 is active with activation 1.5

        // Create a fake loaded encoder
        let clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None],
            config: CltConfig {
                n_layers: 1,
                d_model,
                n_features_per_layer: n_features,
                n_features_total: n_features,
                model_name: "test".to_owned(),
            },
            loaded_encoder: Some(LoadedEncoder {
                layer: 0,
                w_enc,
                b_enc,
            }),
        };

        let sparse = clt.encode(&residual, 0).unwrap();
        assert_eq!(sparse.len(), 1, "only feature 0 should be active");
        assert_eq!(sparse.features[0].0.index, 0);
        assert!((sparse.features[0].1 - 1.5).abs() < 1e-5);
    }

    #[test]
    fn encode_wrong_layer_errors() {
        let device = Device::Cpu;
        let w_enc = Tensor::zeros((4, 8), DType::F32, &device).unwrap();
        let b_enc = Tensor::zeros((4,), DType::F32, &device).unwrap();
        let residual = Tensor::zeros((8,), DType::F32, &device).unwrap();

        let clt = CrossLayerTranscoder {
            repo_id: "test".to_owned(),
            fetch_config: hf_fetch_model::FetchConfig::builder().build().unwrap(),
            encoder_paths: vec![None; 2],
            config: CltConfig {
                n_layers: 2,
                d_model: 8,
                n_features_per_layer: 4,
                n_features_total: 8,
                model_name: "test".to_owned(),
            },
            loaded_encoder: Some(LoadedEncoder {
                layer: 0,
                w_enc,
                b_enc,
            }),
        };

        // Requesting layer 1 when layer 0 is loaded should error.
        let result = clt.encode(&residual, 1);
        assert!(result.is_err());
    }

    #[test]
    fn parse_yaml_value_basic() {
        let yaml = "model_name: \"google/gemma-2-2b\"\nmodel_kind: cross_layer_transcoder\n";
        assert_eq!(
            parse_yaml_value(yaml, "model_name"),
            Some("google/gemma-2-2b".to_owned())
        );
        assert_eq!(
            parse_yaml_value(yaml, "model_kind"),
            Some("cross_layer_transcoder".to_owned())
        );
        assert_eq!(parse_yaml_value(yaml, "missing_key"), None);
    }
}
