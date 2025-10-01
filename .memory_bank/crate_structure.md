# RONN Crate Structure

## Workspace Organization

```
ronn/
├── crates/
│   ├── ronn-core/         # Core tensor operations, session management
│   ├── ronn-providers/    # Execution providers (CPU, GPU, BitNet, WASM, Custom)
│   ├── ronn-onnx/         # ONNX compatibility layer
│   ├── ronn-graph/        # Graph optimization pipeline
│   ├── ronn-hrm/          # Hierarchical reasoning module (System 1/2)
│   ├── ronn-memory/       # Multi-tier memory system (scaffolded)
│   ├── ronn-learning/     # Continual learning engine (scaffolded)
│   └── ronn-api/          # High-level user API
├── examples/
│   ├── simple-inference/  # Basic usage examples
│   ├── brain-features/    # Brain-inspired computing demo
│   └── onnx-model/        # Real ONNX model inference
└── benches/               # Performance benchmarks
```

## Crate Descriptions

### ronn-core (✅ Complete)
**Purpose**: Foundational types and core runtime
**Key Components**:
- Tensor type with Candle integration
- Session lifecycle management (create, run, destroy)
- Data types: F32, F16, BF16, I8, I32, I64, U8, U32, Bool
- Memory layout optimization (row-major, column-major)
- Thread-safe session storage with DashMap

### ronn-providers (✅ Complete)
**Purpose**: Hardware abstraction and execution backends
**Implemented Providers**:
1. **CPU**: SIMD-optimized (AVX2/AVX-512/NEON), Rayon parallelization, NUMA-aware
2. **GPU**: Candle-based CUDA/Metal, multi-GPU support, tensor cores, custom kernels
3. **BitNet**: 1-bit quantization (32x compression), XNOR matmul
4. **WebAssembly**: Browser deployment, SIMD128, IndexedDB caching
5. **Custom**: Plugin framework with NPU/TPU examples

**Key Features**:
- ExecutionProvider trait with capability reporting
- Memory allocator interface with pooling (28.57% hit rate)
- Cross-provider memory transfers
- Multi-GPU topology detection and load balancing

### ronn-onnx (✅ Complete)
**Purpose**: ONNX model loading and operator support
**Key Components**:
- ONNX protobuf parser (loader.rs:31)
- 20 operators in 4 categories:
  - Neural Network (4): Conv2D, BatchNormalization, MaxPool, AveragePool
  - Activation (5): ReLU, Sigmoid, Tanh, Softmax, GELU
  - Mathematical (5): Add, Sub, Mul, Div, MatMul
  - Tensor (6): Reshape, Transpose, Concat, Split, Gather, Slice
- Operator registry with dynamic dispatch (ops/mod.rs:44)
- Type system with 10 data types (types.rs)
- Validation framework for inputs/attributes

### ronn-graph (✅ Complete)
**Purpose**: Graph optimization pipeline
**Optimization Passes** (6 total):
1. Constant folding (passes/constant_folding.rs:8)
2. Dead code elimination (passes/dead_code.rs:8)
3. Node fusion - Conv+BatchNorm+ReLU (passes/fusion.rs:8)
4. Layout optimization - NCHW/NHWC (passes/layout.rs:8)
5. CPU-specific optimizations (passes/provider_specific.rs:7)
6. GPU-specific optimizations (passes/provider_specific.rs:48)

**Optimization Levels**:
- O0: None
- O1: Basic (constant folding, dead code)
- O2: Standard (+ fusion, layout)
- O3: Aggressive (+ provider-specific)

### ronn-hrm (✅ MVP Complete)
**Purpose**: Hierarchical Reasoning Module (brain-inspired)
**Current Implementation** (examples/brain-features/src/main.rs):
- Complexity assessment engine (line 137)
- Input size + variance-based routing (line 154)
- System 1 (Fast): BitNet for simple patterns (line 102)
- System 2 (Slow): Full precision for complex queries (line 102)

**Future Enhancements** (🟡):
- Semantic depth estimation using embeddings
- Novelty detection via similarity matching
- Pattern cache with LRU eviction
- Cognitive technique library (CoT, few-shot, analogical reasoning)
- Meta-cognitive monitoring and replanning

### ronn-memory (🟡 Scaffolded)
**Purpose**: Multi-tier memory system
**Planned Components**:
- Working memory: Short-term, attention-weighted, LRU eviction
- Episodic memory: HNSW vector store, temporal indexing
- Semantic memory: Knowledge graph, concept extraction

### ronn-learning (🟡 Scaffolded)
**Purpose**: Continual learning engine
**Planned Components**:
- Fast/slow weights for multi-timescale adaptation
- Elastic Weight Consolidation (EWC) for catastrophic forgetting prevention
- Experience replay buffer with prioritized sampling
- Meta-learning capabilities

### ronn-api (✅ Complete)
**Purpose**: High-level user-facing API
**Key Components**:
- Model loading with builder pattern (model.rs)
- Session creation with configuration (session.rs)
- Error handling with structured types (error.rs)
- Synchronous inference support
- Provider selection and optimization level configuration

**Future Enhancements** (🟡):
- Asynchronous inference
- Batch processing support

## Key Dependencies
- **Candle**: Core tensor ops and GPU acceleration
- **Tokio**: Async runtime for background processing
- **Rayon**: Data parallelism for CPU operations
- **DashMap**: Concurrent hash maps for shared state
- **HNSW**: Vector similarity search (for memory systems)
- **Criterion**: Benchmarking framework
