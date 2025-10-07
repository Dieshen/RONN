use crate::common::*;
/// Integration tests for GPT-2 Small text generation model.
///
/// Tests:
/// - Model loading and metadata validation
/// - Single-step inference (next token prediction)
/// - Multi-step autoregressive generation
/// - Accuracy validation against HuggingFace transformers
/// - Performance benchmarks
/// - Greedy vs sampling decoding
use std::time::Instant;

const MODEL_NAME: &str = "gpt2-small.onnx";

/// Test that GPT-2 Small model can be loaded successfully
#[test]
#[ignore] // Remove once model is exported
fn test_gpt2_load() {
    if !model_exists(MODEL_NAME) {
        eprintln!(
            "Model not found: {}. Run 'cd models && python export_gpt2.py'",
            MODEL_NAME
        );
        return;
    }

    let model_path = model_path(MODEL_NAME);
    println!("Loading GPT-2 Small from: {}", model_path);

    let model = ronn_onnx::ModelLoader::load_from_file(&model_path)
        .expect("Failed to load GPT-2 Small model");

    // Validate model metadata
    println!("GPT-2 Small loaded successfully!");
    println!("  IR version: {}", model.ir_version);
    println!("  Producer: {:?}", model.producer_name);
    println!("  Inputs: {}", model.inputs().len());
    println!("  Outputs: {}", model.outputs().len());

    // GPT-2 typically has 1 input (input_ids) and 1 output (hidden states or logits)
    assert_eq!(
        model.inputs().len(),
        1,
        "GPT-2 should have 1 input (input_ids)"
    );
    assert_eq!(
        model.outputs().len(),
        1,
        "GPT-2 should have 1 output (last_hidden_state or logits)"
    );

    // Validate input
    let input = &model.inputs()[0];
    println!(
        "  Input: {} {:?} {:?}",
        input.name, input.shape, input.data_type
    );

    // Input should be 2D: [batch, sequence_length]
    assert_eq!(
        input.shape.len(),
        2,
        "Input should have 2 dimensions [batch, sequence]"
    );

    // Validate output
    let output = &model.outputs()[0];
    println!(
        "  Output: {} {:?} {:?}",
        output.name, output.shape, output.data_type
    );

    // Output should be 3D: [batch, sequence_length, hidden_size]
    // GPT-2 Small has hidden_size = 768
    assert_eq!(
        output.shape.len(),
        3,
        "Output should have 3 dimensions [batch, sequence, hidden/vocab]"
    );
}

/// Test GPT-2 single-step inference (next token prediction)
#[test]
#[ignore]
fn test_gpt2_single_step_inference() {
    if !model_exists(MODEL_NAME) {
        eprintln!("Model not found: {}", MODEL_NAME);
        return;
    }

    let model_path = model_path(MODEL_NAME);
    let model = ronn_onnx::ModelLoader::load_from_file(&model_path).expect("Failed to load GPT-2");

    println!("\nRunning single-step inference (next token prediction)...");

    // Sample prompt: "Once upon a time" tokenized
    // In real usage, use GPT-2 tokenizer to get token IDs
    // For testing, use dummy token IDs
    let input_ids: Vec<i64> = vec![5962, 2402, 257, 640]; // "Once upon a time" (example IDs)

    println!("  Input sequence length: {}", input_ids.len());
    println!("  Input token IDs: {:?}", input_ids);

    // TODO: Once ronn-api Session supports INT64 inputs and autoregressive inference:
    //
    // let session = model.create_session_default()?;
    // let mut inputs = HashMap::new();
    //
    // let input_ids_tensor = Tensor::from_data(
    //     input_ids.clone(),
    //     vec![1, input_ids.len()],
    //     DataType::I64,
    //     TensorLayout::RowMajor,
    // )?;
    //
    // inputs.insert("input_ids".to_string(), input_ids_tensor);
    //
    // let outputs = session.run(inputs)?;
    //
    // // Get hidden states: [1, sequence_length, 768]
    // let hidden_states = outputs.get("last_hidden_state").unwrap();
    // assert_eq!(hidden_states.shape()[0], 1);
    // assert_eq!(hidden_states.shape()[1], input_ids.len());
    // assert_eq!(hidden_states.shape()[2], 768);
    //
    // // To get next token logits, need to:
    // // 1. Take last hidden state: hidden_states[:, -1, :]
    // // 2. Project to vocabulary: matmul with lm_head weights (768 -> vocab_size)
    // // 3. Apply softmax or argmax for next token

    println!("  ✓ Single-step inference completed (placeholder)");
    println!("  Note: Full inference requires Session integration");
}

/// Test GPT-2 multi-step autoregressive generation
#[test]
#[ignore]
fn test_gpt2_autoregressive_generation() {
    if !model_exists(MODEL_NAME) {
        eprintln!("Model not found: {}", MODEL_NAME);
        return;
    }

    println!("\nTesting GPT-2 autoregressive text generation...");

    // This test implements the full autoregressive generation loop:
    // 1. Start with prompt tokens
    // 2. Run inference to get next token logits
    // 3. Sample/select next token
    // 4. Append to sequence
    // 5. Repeat for N steps

    const MAX_NEW_TOKENS: usize = 20;
    let initial_prompt = vec![5962i64, 2402, 257, 640]; // "Once upon a time"

    println!("  Initial prompt length: {}", initial_prompt.len());
    println!("  Generating {} new tokens", MAX_NEW_TOKENS);

    // TODO: Implement autoregressive generation loop
    //
    // let model = ModelLoader::load_from_file(model_path(MODEL_NAME))?;
    // let session = model.create_session_default()?;
    //
    // let mut generated_tokens = initial_prompt.clone();
    //
    // for step in 0..MAX_NEW_TOKENS {
    //     // Run inference with current sequence
    //     let outputs = session.run(create_input(&generated_tokens))?;
    //     let hidden_states = outputs.get("last_hidden_state")?;
    //
    //     // Get last token's hidden state
    //     let last_hidden = extract_last_token_hidden(hidden_states);
    //
    //     // Project to vocabulary (requires lm_head weights)
    //     let logits = project_to_vocab(last_hidden);
    //
    //     // Select next token (greedy: argmax)
    //     let next_token_id = argmax(&logits);
    //
    //     // Append to sequence
    //     generated_tokens.push(next_token_id);
    //
    //     println!("  Step {}: token_id={}", step, next_token_id);
    //
    //     // Stop if EOS token
    //     if next_token_id == 50256 { // GPT-2 EOS token
    //         break;
    //     }
    // }
    //
    // println!("  Final sequence length: {}", generated_tokens.len());

    println!("  Autoregressive generation - TODO");
    println!("  Requires: Session API + LM head projection layer");
}

/// Test GPT-2 greedy vs sampling decoding strategies
#[test]
#[ignore]
fn test_gpt2_decoding_strategies() {
    if !model_exists(MODEL_NAME) {
        eprintln!("Model not found: {}", MODEL_NAME);
        return;
    }

    println!("\nTesting different decoding strategies...");

    // Compare different decoding strategies:
    // 1. Greedy decoding (argmax)
    // 2. Top-k sampling
    // 3. Top-p (nucleus) sampling
    // 4. Temperature sampling

    // Each strategy should produce different but coherent outputs

    println!("  Decoding strategies - TODO");
    println!("  Strategies: Greedy, Top-k, Top-p, Temperature");
}

/// Test GPT-2 accuracy against HuggingFace reference
#[test]
#[ignore]
fn test_gpt2_accuracy_reference() {
    if !model_exists(MODEL_NAME) {
        eprintln!("Model not found: {}", MODEL_NAME);
        return;
    }

    println!("\nValidating GPT-2 accuracy against HuggingFace reference...");

    // This test would:
    // 1. Use a known input sequence
    // 2. Run inference in RONN
    // 3. Compare hidden states against HuggingFace GPT2Model output
    // 4. Verify numerical accuracy within tolerance

    // Reference outputs should be generated with:
    // from transformers import GPT2Model, GPT2Tokenizer
    // tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    // model = GPT2Model.from_pretrained("gpt2")
    // inputs = tokenizer("Sample text", return_tensors="pt")
    // outputs = model(**inputs)
    // np.save("gpt2_reference.npy", outputs.last_hidden_state.detach().numpy())

    // Example comparison:
    // let reference_path = reference_output_path("gpt2_sample1");
    // let expected_hidden_states = load_reference_output(&reference_path);
    // let mae = mean_absolute_error(&actual_hidden_states, &expected_hidden_states);
    // assert!(mae < FP32_TOLERANCE, "MAE {} exceeds tolerance", mae);

    println!("  Reference accuracy validation - TODO");
    println!("  Requires: HuggingFace transformers reference output");
}

/// Benchmark GPT-2 inference performance
#[test]
#[ignore]
fn test_gpt2_performance() {
    if !model_exists(MODEL_NAME) {
        eprintln!("Model not found: {}", MODEL_NAME);
        return;
    }

    println!("\nBenchmarking GPT-2 inference performance...");

    const WARMUP_ITERATIONS: usize = 10;
    const BENCHMARK_ITERATIONS: usize = 100;
    const SEQUENCE_LENGTH: usize = 64;

    // TODO: Implement performance benchmarking
    //
    // let model = ModelLoader::load_from_file(model_path(MODEL_NAME))?;
    // let session = model.create_session_default()?;
    //
    // let input_ids = create_dummy_token_ids(1, SEQUENCE_LENGTH);
    //
    // // Warmup
    // for _ in 0..WARMUP_ITERATIONS {
    //     let _ = session.run(create_input(&input_ids));
    // }
    //
    // // Benchmark
    // let mut latencies = Vec::with_capacity(BENCHMARK_ITERATIONS);
    // for _ in 0..BENCHMARK_ITERATIONS {
    //     let start = Instant::now();
    //     let _ = session.run(create_input(&input_ids));
    //     let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    //     latencies.push(elapsed);
    // }
    //
    // let metrics = PerformanceMetrics::from_sorted_latencies(latencies);
    // metrics.print();

    println!("  Performance benchmarking - TODO");
    println!("  Sequence length: {}", SEQUENCE_LENGTH);
}

/// Test GPT-2 with different sequence lengths
#[test]
#[ignore]
fn test_gpt2_variable_sequence_lengths() {
    if !model_exists(MODEL_NAME) {
        eprintln!("Model not found: {}", MODEL_NAME);
        return;
    }

    println!("\nTesting GPT-2 with variable sequence lengths...");

    // Test with different sequence lengths: 16, 32, 64, 128, 256, 512, 1024
    // GPT-2 has a maximum context length of 1024 tokens

    const SEQUENCE_LENGTHS: &[usize] = &[16, 32, 64, 128, 256, 512, 1024];

    for &seq_len in SEQUENCE_LENGTHS {
        println!("  Sequence length: {}", seq_len);

        // TODO: Run inference with different sequence lengths
        // Verify output shape: [1, seq_len, 768]
        // Measure inference time vs sequence length
        // Expect O(n²) scaling due to self-attention
    }

    println!("  Variable sequence length testing - TODO");
}

/// Test GPT-2 batch inference for throughput
#[test]
#[ignore]
fn test_gpt2_batch_inference() {
    if !model_exists(MODEL_NAME) {
        eprintln!("Model not found: {}", MODEL_NAME);
        return;
    }

    println!("\nTesting GPT-2 batch inference...");

    // Test with batch sizes: 1, 4, 8, 16
    // Measure throughput (tokens/sec)

    const BATCH_SIZES: &[usize] = &[1, 4, 8, 16];
    const SEQUENCE_LENGTH: usize = 64;

    for &batch_size in BATCH_SIZES {
        println!("  Batch size: {}", batch_size);

        // TODO: Implement batch inference
        // Create input tensors with shape [batch_size, SEQUENCE_LENGTH]
        // Run inference
        // Measure throughput (tokens generated per second)
    }

    println!("  Batch inference testing - TODO");
}

/// Test GPT-2 KV cache optimization (if implemented)
#[test]
#[ignore]
fn test_gpt2_kv_cache() {
    if !model_exists(MODEL_NAME) {
        eprintln!("Model not found: {}", MODEL_NAME);
        return;
    }

    println!("\nTesting GPT-2 with KV cache optimization...");

    // KV cache stores past key-value pairs from previous tokens
    // to avoid recomputing them during autoregressive generation

    // Compare performance:
    // 1. Without KV cache: full re-computation each step
    // 2. With KV cache: incremental computation

    // Expected speedup: 3-5x for long sequences

    println!("  KV cache optimization - TODO");
    println!("  Expected speedup: 3-5x for autoregressive generation");
}
