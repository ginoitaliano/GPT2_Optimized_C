# GPT-2 C Implementation

## Overview

This repository contains a high-performance C implementation of the GPT-2 transformer model. The implementation focuses on efficiency through multi-threading and AVX instruction utilization while maintaining the core architecture of the original model.

## Features

- Complete implementation of the GPT-2 transformer architecture
- Multi-threaded attention mechanism for improved performance
- AVX instruction set utilization for accelerated matrix operations
- Memory-efficient design for handling large models
- Support for the 124M parameter GPT-2 model (12 layers, 12 heads, 768 embedding dimensions)

## Technical Specifications

- Embedding size: 768
- Number of transformer blocks: 12
- Number of attention heads: 12
- Head dimension: 64 (768/12)
- Vocabulary size: 50,257
- Maximum position embeddings: 1,024
- Maximum threads for parallelization: 8 (configurable)

## Requirements

- GCC or Clang compiler with C11 support
- POSIX threads library (pthread)
- AVX instruction set support (Intel processors since Sandy Bridge or AMD processors since Bulldozer)
- Math library (`-lm`)

## Building

To compile the program:

```bash
gcc -O3 -mavx -pthread -lm -o gpt2 gpt2.c
```

For maximum performance:

```bash
gcc -O3 -march=native -mavx2 -ffast-math -pthread -lm -o gpt2 gpt2.c
```

## Usage

```bash
./gpt2
```

The default implementation initializes random weights and processes a sample input. For practical use, you'll need to:

1. Load pre-trained weights from a file
2. Implement tokenization for input text
3. Add temperature-based sampling for text generation

## Code Structure

- **Linear Layer Implementation**: Matrix multiplication with bias addition
- **Attention Mechanism**: 
  - Multi-headed scaled dot-product attention
  - Parallelized implementation using pthreads
  - AVX-accelerated dot products
- **Layer Normalization**: For stabilizing network activations
- **Activation Functions**: GELU (Gaussian Error Linear Unit)
- **Memory Management**: Comprehensive cleanup functions to prevent leaks

## Extending the Code

### Loading Pre-trained Weights

To load actual GPT-2 weights, implement a function to read weights from a file:

```c
GPT2Weights load_weights_from_file(const char* filename) {
    GPT2Weights weights;
    FILE* file = fopen(filename, "rb");
    // Read weights from file
    // ...
    fclose(file);
    return weights;
}
```

### Text Generation

Implement sampling from logits for text generation:

```c
int sample_token(float* logits, float temperature) {
    // Apply temperature
    // ...
    // Sample from distribution
    // ...
    return sampled_token_id;
}
```

### Tokenization

Add functions to convert between text and tokens:

```c
int* tokenize(const char* text, int* length) {
    // Implement tokenization
    // ...
    return tokens;
}

char* detokenize(int* tokens, int length) {
    // Convert tokens back to text
    // ...
    return text;
}
```

## Performance Optimization

The code includes several optimizations:

1. **Multi-threading**: The attention mechanism is parallelized across multiple threads
2. **SIMD Instructions**: AVX instructions are used for fast vector operations
3. **Memory Efficiency**: Careful memory management to minimize allocations

## Memory Requirements

For the 124M parameter model:
- Word token embeddings: ~154MB (50,257 * 768 * 4 bytes)
- Position embeddings: ~3MB (1,024 * 768 * 4 bytes)
- Transformer blocks: ~285MB (12 blocks * parameters per block)
- Total: ~450MB for model parameters

## License

[MIT License](LICENSE)

## References

- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Transformer Architecture](https://arxiv.org/abs/1706.03762)
- [OpenAI GPT-2 Repository](https://github.com/openai/gpt-2)
