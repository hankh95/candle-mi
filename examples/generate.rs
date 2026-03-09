// SPDX-License-Identifier: MIT OR Apache-2.0

//! Autoregressive text generation with greedy decoding.
//!
//! ```bash
//! cargo run --release --features transformer --example generate
//! ```
//!
//! **What it does:**
//!
//! 1. Scans the local `HuggingFace` Hub cache for supported transformers.
//! 2. For each cached model, tokenizes the prompt *"The capital of France
//!    is"* and runs a greedy autoregressive generation loop (temperature 0)
//!    for up to 20 tokens, printing each token as it is produced.
//! 3. Builds a [`GenerationResult`](candle_mi::GenerationResult) and
//!    prints a summary.
//!
//! The forward pass recomputes the full sequence at every step (no KV
//! cache).  This is intentional: candle-mi prioritises interpretability
//! (all activations available for analysis) over inference speed.
//!
//! Each model is dropped before the next one loads, so GPU memory is
//! reused.

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::unnecessary_wraps)]

use candle_mi::{
    GenerationResult, HookSpec, MIModel, MITokenizer, SUPPORTED_MODEL_TYPES, sample_token,
};
use std::path::{Path, PathBuf};

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> candle_mi::Result<()> {
    let prompt = "The capital of France is";
    let max_new_tokens: usize = 20;

    // 1. Discover cached models
    let cached = discover_cached_models();
    if cached.is_empty() {
        println!("No cached transformer models found in the HuggingFace Hub cache.");
        println!("Download one first, e.g.:");
        println!("  cargo run --example fast_download -- meta-llama/Llama-3.2-1B");
        return Ok(());
    }

    println!(
        "Found {} supported transformer(s) in HF cache:\n",
        cached.len()
    );

    // 2. Generate with each model
    for (model_id, model_type, snapshot) in &cached {
        println!("=== {model_id} (model_type: {model_type}) ===");
        if let Err(e) = run_model(model_id, snapshot, prompt, max_new_tokens) {
            println!("  Skipped: {e}\n");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Cache discovery (same pattern as quick_start_transformer)
// ---------------------------------------------------------------------------

/// Return the `HuggingFace` Hub cache directory.
fn hf_cache_dir() -> Option<PathBuf> {
    if let Ok(cache) = std::env::var("HF_HOME") {
        return Some(PathBuf::from(cache).join("hub"));
    }
    if let Ok(home) = std::env::var("USERPROFILE") {
        let p = PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
        if p.is_dir() {
            return Some(p);
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        let p = PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
        if p.is_dir() {
            return Some(p);
        }
    }
    None
}

/// Find the first snapshot directory for a cached model.
fn find_snapshot(cache_dir: &Path, model_id: &str) -> Option<PathBuf> {
    let dir_name = format!("models--{}", model_id.replace('/', "--"));
    let snapshots = cache_dir.join(dir_name).join("snapshots");
    let entry = std::fs::read_dir(snapshots).ok()?.next()?.ok()?;
    Some(entry.path())
}

/// Read `model_type` from a cached `config.json`.
fn read_model_type(snapshot: &Path) -> Option<String> {
    let config_path = snapshot.join("config.json");
    let text = std::fs::read_to_string(config_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    // BORROW: explicit .as_str() — serde_json::Value → &str
    json.get("model_type")?.as_str().map(String::from)
}

/// Scan the HF cache and return `(model_id, model_type, snapshot_path)`.
fn discover_cached_models() -> Vec<(String, String, PathBuf)> {
    let Some(cache_dir) = hf_cache_dir() else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(&cache_dir) else {
        return Vec::new();
    };

    let mut models = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(dir_name) = name.to_str() else {
            continue;
        };
        let Some(rest) = dir_name.strip_prefix("models--") else {
            continue;
        };
        let model_id = rest.replacen("--", "/", 1);
        let Some(snapshot) = find_snapshot(&cache_dir, &model_id) else {
            continue;
        };
        let Some(model_type) = read_model_type(&snapshot) else {
            continue;
        };
        // BORROW: explicit .as_str() — String → &str for slice lookup
        if SUPPORTED_MODEL_TYPES.contains(&model_type.as_str()) {
            models.push((model_id, model_type, snapshot));
        }
    }
    models.sort_by(|a, b| a.0.cmp(&b.0));
    models
}

// ---------------------------------------------------------------------------
// Per-model generation
// ---------------------------------------------------------------------------

/// Load a model, generate tokens, and print results.
fn run_model(
    model_id: &str,
    snapshot: &Path,
    prompt: &str,
    max_new_tokens: usize,
) -> candle_mi::Result<()> {
    let model = MIModel::from_pretrained(model_id)?;
    println!(
        "  {} layers, {} hidden, device: {:?}",
        model.num_layers(),
        model.hidden_size(),
        model.device()
    );

    let tokenizer_path = snapshot.join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Err(candle_mi::MIError::Tokenizer(
            "tokenizer.json not found in snapshot".into(),
        ));
    }
    let tokenizer = MITokenizer::from_hf_path(tokenizer_path)?;

    let prompt_tokens = tokenizer.encode(prompt)?;

    let result = generate(&model, &tokenizer, &prompt_tokens, max_new_tokens)?;

    println!("\n  --- Generation Result ---");
    println!("  Prompt tokens : {}", result.prompt_tokens.len());
    println!("  Generated     : {}", result.generated_tokens.len());
    println!("  Full text     : \"{}\"", result.full_text);
    println!();

    Ok(())
}

// ---------------------------------------------------------------------------
// Generation loop
// ---------------------------------------------------------------------------

/// Run greedy autoregressive generation.
///
/// Recomputes the full sequence at every step so that all intermediate
/// activations remain available for mechanistic interpretability analyses.
fn generate(
    model: &MIModel,
    tokenizer: &MITokenizer,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
) -> candle_mi::Result<GenerationResult> {
    let mut tokens = prompt_tokens.to_vec();
    let hooks = HookSpec::new();
    let prompt_text = tokenizer.decode(prompt_tokens)?;

    print!("  {prompt_text}");

    for _ in 0..max_new_tokens {
        // Build input: full sequence each step — shape [1, seq_len]
        let input = candle_core::Tensor::new(tokens.as_slice(), model.device())?.unsqueeze(0)?;

        // Forward pass (empty hooks = zero overhead)
        let cache = model.forward(&input, &hooks)?;
        let logits = cache.output(); // [1, seq_len, vocab]

        // Extract last-token logits — shape [vocab]
        let seq_len = tokens.len();
        let last_logits = logits.get(0)?.get(seq_len - 1)?;

        // Greedy decode (temperature 0)
        let next_token = sample_token(&last_logits, 0.0)?;

        // Print token as it is produced
        let token_text = tokenizer.decode(&[next_token])?;
        print!("{token_text}");

        tokens.push(next_token);
    }

    println!();

    // Build result
    let prompt_len = prompt_tokens.len();
    let generated_tokens = tokens.get(prompt_len..).unwrap_or_default().to_vec();
    let full_text = tokenizer.decode(&tokens)?;
    let generated_text = tokenizer.decode(&generated_tokens)?;

    Ok(GenerationResult {
        prompt: prompt_text,
        full_text,
        generated_text,
        prompt_tokens: prompt_tokens.to_vec(),
        generated_tokens,
        total_tokens: tokens.len(),
    })
}
