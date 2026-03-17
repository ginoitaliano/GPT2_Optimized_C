#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <immintrin.h>  // For AVX instructions

#define EPSILON 1e-5
#define EMBEDDING_SIZE 768
#define NUM_BLOCKS 12
#define NUM_HEADS 12
#define HEAD_DIM (EMBEDDING_SIZE / NUM_HEADS)
#define VOCAB_SIZE 50257
#define MAX_POSITION_EMBEDDINGS 1024
#define MAX_THREADS 8  // Configurable thread count

// Linear Layer Structure
typedef struct {
    float **weights;
    float *biases;
    int fcInputSize;
    int fcOutputSize;
} LinearLayer;

// Block Weights Structure
typedef struct {
    LinearLayer q_mlp;
    LinearLayer k_mlp;
    LinearLayer v_mlp;
    LinearLayer first_block_MLP;
    LinearLayer second_block_MLP;
} BlockWeights;

// GPT-2 Model Weights Structure
typedef struct {
    float **wte;  // Word token embeddings
    float **wpe;  // Position embeddings
    BlockWeights *blocks;
    LinearLayer logits_mlp;
} GPT2Weights;

// Attention Thread Data Structure
typedef struct {
    float **Q;
    float **K;
    float **V;
    float **output;
    int seqLength;
    int depth;
    int threadId;
} AttentionThreadData;

// Prototype Declarations
float *linear(float *input, float **weights, float *biases, int inputSize, int outputSize);
float **scaled_dot_product_attention(float **Q, float **K, float **V, int seqLength, int depth);
float **matrix_add(float **x, float **y, int numRow, int numCol);
float **norm(float **x, int seqLength, int features);
float *gelu(float *x, int size);
int *positions_for(int *tokens, int seqLength, int past_length);

// Linear Layer Computation
float *linear(float *input, float **weights, float *biases, int inputSize, int outputSize) {
    float *output = (float *)malloc(outputSize * sizeof(float));
    for (int i = 0; i < outputSize; i++) {
        output[i] = biases[i];
        for (int j = 0; j < inputSize; j++) {
            output[i] += input[j] * weights[i][j];
        }
    }
    return output;
}

// Multi-threaded Scaled Dot Product Attention
void *parallel_scaled_dot_product_attention(void *arg) {
    AttentionThreadData *data = (AttentionThreadData *)arg;
    int start_row = data->threadId * (data->seqLength / MAX_THREADS);
    int end_row = (data->threadId == MAX_THREADS - 1)
                  ? data->seqLength
                  : (data->threadId + 1) * (data->seqLength / MAX_THREADS);

    for (int i = start_row; i < end_row; i++) {
        data->output[i] = (float *)malloc(data->depth * sizeof(float));
        memset(data->output[i], 0, data->depth * sizeof(float));

        for (int k = 0; k < data->depth; k++) {
            float sum = 0.0;
            for (int j = 0; j < data->seqLength; j++) {
                sum += data->Q[i][k] * data->V[j][k];
            }
            data->output[i][k] = sum;
        }
    }
    return NULL;
}

// Parallel Dot Product Attention
float **parallel_dot_product_attention(float **Q, float **K, float **V, int seqLength, int depth) {
    pthread_t threads[MAX_THREADS];
    AttentionThreadData threadData[MAX_THREADS];
    float **output = (float **)malloc(seqLength * sizeof(float *));

    for (int t = 0; t < MAX_THREADS; t++) {
        threadData[t].Q = Q;
        threadData[t].K = K;
        threadData[t].V = V;
        threadData[t].output = output;
        threadData[t].seqLength = seqLength;
        threadData[t].depth = depth;
        threadData[t].threadId = t;
    }

    for (int t = 0; t < MAX_THREADS; t++) {
        pthread_create(&threads[t], NULL, parallel_scaled_dot_product_attention, &threadData[t]);
    }

    for (int t = 0; t < MAX_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

    return output;
}

// Scaled Dot Product Attention
float **scaled_dot_product_attention(float **Q, float **K, float **V, int seqLength, int depth) {
    float **scores = (float **)malloc(seqLength * sizeof(float *));
    for (int i = 0; i < seqLength; i++) {
        scores[i] = (float *)malloc(seqLength * sizeof(float));
        for (int j = 0; j < seqLength; j++) {
            float sum = 0.0;
            for (int k = 0; k < depth; k++) {
                sum += Q[i][k] * K[j][k];
            }
            scores[i][j] = sum / sqrt((float)depth);
        }
    }

    float **attention_weights = (float **)malloc(seqLength * sizeof(float *));
    for (int i = 0; i < seqLength; i++) {
        attention_weights[i] = (float *)malloc(seqLength * sizeof(float));
        float sum_exp = 0.0;
        for (int j = 0; j < seqLength; j++) {
            attention_weights[i][j] = exp(scores[i][j]);
            sum_exp += attention_weights[i][j];
        }
        for (int j = 0; j < seqLength; j++) {
            attention_weights[i][j] /= sum_exp;
        }
    }

    float **output = (float **)malloc(seqLength * sizeof(float *));
    for (int i = 0; i < seqLength; i++) {
        output[i] = (float *)malloc(depth * sizeof(float));
        for (int k = 0; k < depth; k++) {
            output[i][k] = 0.0;
            for (int j = 0; j < seqLength; j++) {
                output[i][k] += attention_weights[i][j] * V[j][k];
            }
        }
    }

    for (int i = 0; i < seqLength; i++) {
        free(scores[i]);
        free(attention_weights[i]);
    }
    free(scores);
    free(attention_weights);

    return output;
}

// Matrix Addition
float **matrix_add(float **x, float **y, int numRow, int numCol) {
    float **result = (float **)malloc(numRow * sizeof(float *));
    for (int i = 0; i < numRow; i++) {
        result[i] = (float *)malloc(numCol * sizeof(float));
        for (int j = 0; j < numCol; j++) {
            result[i][j] = x[i][j] + y[i][j];
        }
    }
    return result;
}

// Layer Normalization
float **norm(float **x, int seqLength, int features) {
    float **normalized = (float **)malloc(seqLength * sizeof(float *));
    for (int i = 0; i < seqLength; i++) {
        normalized[i] = (float *)malloc(features * sizeof(float));
        float mean = 0.0;
        for (int j = 0; j < features; j++) {
            mean += x[i][j];
        }
        mean /= features;

        float variance = 0.0;
        for (int j = 0; j < features; j++) {
            variance += (x[i][j] - mean) * (x[i][j] - mean);
        }
        variance /= features;

        for (int j = 0; j < features; j++) {
            normalized[i][j] = (x[i][j] - mean) / sqrt(variance + EPSILON);
        }
    }
    return normalized;
}

// GELU Activation Function
float *gelu(float *x, int size) {
    float *output = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        output[i] = 0.5f * x[i] * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x[i] + 0.044715f * x[i] * x[i] * x[i])));
    }
    return output;
}

// Position Computation
int *positions_for(int *tokens, int seqLength, int past_length) {
    int *positions = (int *)malloc(seqLength * sizeof(int));
    for (int i = 0; i < seqLength; i++) {
        positions[i] = past_length + i;
    }
    return positions;
}

// Helper: initialize a LinearLayer with random weights
static LinearLayer init_linear(int inputSize, int outputSize) {
    LinearLayer layer;
    layer.fcInputSize  = inputSize;
    layer.fcOutputSize = outputSize;
    layer.weights = (float **)malloc(outputSize * sizeof(float *));
    for (int i = 0; i < outputSize; i++) {
        layer.weights[i] = (float *)malloc(inputSize * sizeof(float));
        for (int j = 0; j < inputSize; j++) {
            layer.weights[i][j] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
        }
    }
    layer.biases = (float *)malloc(outputSize * sizeof(float));
    for (int i = 0; i < outputSize; i++) {
        layer.biases[i] = 0.0f;
    }
    return layer;
}

// Transformer Block
float **block(float **x, int seqLength, int embeddingSize, BlockWeights weights) {
    float **normalized_x = norm(x, seqLength, embeddingSize);

    float **Q = (float **)malloc(seqLength * sizeof(float *));
    float **K = (float **)malloc(seqLength * sizeof(float *));
    float **V = (float **)malloc(seqLength * sizeof(float *));
    for (int i = 0; i < seqLength; i++) {
        Q[i] = linear(normalized_x[i], weights.q_mlp.weights, weights.q_mlp.biases,
                      weights.q_mlp.fcInputSize, weights.q_mlp.fcOutputSize);
        K[i] = linear(normalized_x[i], weights.k_mlp.weights, weights.k_mlp.biases,
                      weights.k_mlp.fcInputSize, weights.k_mlp.fcOutputSize);
        V[i] = linear(normalized_x[i], weights.v_mlp.weights, weights.v_mlp.biases,
                      weights.v_mlp.fcInputSize, weights.v_mlp.fcOutputSize);
    }

    // Free normalized_x
    for (int i = 0; i < seqLength; i++) free(normalized_x[i]);
    free(normalized_x);

    float ***Q_heads    = (float ***)malloc(NUM_HEADS * sizeof(float **));
    float ***K_heads    = (float ***)malloc(NUM_HEADS * sizeof(float **));
    float ***V_heads    = (float ***)malloc(NUM_HEADS * sizeof(float **));
    float ***head_outputs = (float ***)malloc(NUM_HEADS * sizeof(float **));

    for (int h = 0; h < NUM_HEADS; h++) {
        Q_heads[h] = (float **)malloc(seqLength * sizeof(float *));
        K_heads[h] = (float **)malloc(seqLength * sizeof(float *));
        V_heads[h] = (float **)malloc(seqLength * sizeof(float *));

        for (int i = 0; i < seqLength; i++) {
            Q_heads[h][i] = (float *)malloc(HEAD_DIM * sizeof(float));
            K_heads[h][i] = (float *)malloc(HEAD_DIM * sizeof(float));
            V_heads[h][i] = (float *)malloc(HEAD_DIM * sizeof(float));
            memcpy(Q_heads[h][i], &Q[i][h * HEAD_DIM], HEAD_DIM * sizeof(float));
            memcpy(K_heads[h][i], &K[i][h * HEAD_DIM], HEAD_DIM * sizeof(float));
            memcpy(V_heads[h][i], &V[i][h * HEAD_DIM], HEAD_DIM * sizeof(float));
        }

        head_outputs[h] = scaled_dot_product_attention(
            Q_heads[h], K_heads[h], V_heads[h], seqLength, HEAD_DIM);

        // Free head inputs
        for (int i = 0; i < seqLength; i++) {
            free(Q_heads[h][i]);
            free(K_heads[h][i]);
            free(V_heads[h][i]);
        }
        free(Q_heads[h]);
        free(K_heads[h]);
        free(V_heads[h]);
    }
    free(Q_heads);
    free(K_heads);
    free(V_heads);

    // Free Q, K, V
    for (int i = 0; i < seqLength; i++) { free(Q[i]); free(K[i]); free(V[i]); }
    free(Q); free(K); free(V);

    // Concatenate head outputs
    float **a = (float **)malloc(seqLength * sizeof(float *));
    for (int i = 0; i < seqLength; i++) {
        a[i] = (float *)malloc(embeddingSize * sizeof(float));
        for (int h = 0; h < NUM_HEADS; h++) {
            memcpy(&a[i][h * HEAD_DIM], head_outputs[h][i], HEAD_DIM * sizeof(float));
            free(head_outputs[h][i]);
        }
    }
    for (int h = 0; h < NUM_HEADS; h++) free(head_outputs[h]);
    free(head_outputs);

    // Residual connection
    float **x_added = matrix_add(x, a, seqLength, embeddingSize);
    for (int i = 0; i < seqLength; i++) free(a[i]);
    free(a);

    float **normalized_x_added = norm(x_added, seqLength, embeddingSize);

    float **m = (float **)malloc(seqLength * sizeof(float *));
    for (int i = 0; i < seqLength; i++) {
        float *first_mlp_output = linear(normalized_x_added[i],
            weights.first_block_MLP.weights, weights.first_block_MLP.biases,
            weights.first_block_MLP.fcInputSize, weights.first_block_MLP.fcOutputSize);

        float *gelu_output = gelu(first_mlp_output, weights.first_block_MLP.fcOutputSize);
        free(first_mlp_output);

        m[i] = linear(gelu_output,
            weights.second_block_MLP.weights, weights.second_block_MLP.biases,
            weights.second_block_MLP.fcInputSize, weights.second_block_MLP.fcOutputSize);
        free(gelu_output);
    }

    for (int i = 0; i < seqLength; i++) free(normalized_x_added[i]);
    free(normalized_x_added);

    float **output = matrix_add(x_added, m, seqLength, embeddingSize);

    for (int i = 0; i < seqLength; i++) { free(x_added[i]); free(m[i]); }
    free(x_added);
    free(m);

    return output;
}

// GPT-2 Model
float *model(int *tokens, int seqLength, GPT2Weights weights) {
    int past_length = 0;
    int *positions = positions_for(tokens, seqLength, past_length);

    float **h = (float **)malloc(seqLength * sizeof(float *));
    for (int i = 0; i < seqLength; i++) {
        h[i] = (float *)malloc(EMBEDDING_SIZE * sizeof(float));
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
    
            h[i][j] = weights.wte[tokens[i]][j] + weights.wpe[positions[i]][j];
        }
    }

    free(positions);

    for (int i = 0; i < NUM_BLOCKS; i++) {
        float **new_h = block(h, seqLength, EMBEDDING_SIZE, weights.blocks[i]);
        for (int j = 0; j < seqLength; j++) free(h[j]);
        free(h);
        h = new_h;
    }

    float *logits = (float *)malloc(VOCAB_SIZE * sizeof(float));
    for (int i = 0; i < VOCAB_SIZE; i++) {
        logits[i] = weights.logits_mlp.biases[i];
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
            logits[i] += h[seqLength - 1][j] * weights.logits_mlp.weights[i][j];
        }
    }

    for (int j = 0; j < seqLength; j++) free(h[j]);
    free(h);

    return logits;
}

// Memory Cleanup Functions
void free_linear_layer(LinearLayer layer) {
    for (int i = 0; i < layer.fcOutputSize; i++) free(layer.weights[i]);
    free(layer.weights);
    free(layer.biases);
}

void free_block_weights(BlockWeights bw) {
    free_linear_layer(bw.q_mlp);
    free_linear_layer(bw.k_mlp);
    free_linear_layer(bw.v_mlp);
    free_linear_layer(bw.first_block_MLP);
    free_linear_layer(bw.second_block_MLP);
}

void free_gpt2_weights(GPT2Weights weights) {
    for (int i = 0; i < VOCAB_SIZE; i++) free(weights.wte[i]);
    free(weights.wte);

    for (int i = 0; i < MAX_POSITION_EMBEDDINGS; i++) free(weights.wpe[i]);
    free(weights.wpe);

    for (int i = 0; i < NUM_BLOCKS; i++) free_block_weights(weights.blocks[i]);
    free(weights.blocks);

    free_linear_layer(weights.logits_mlp);
}

// Weight Initialization
GPT2Weights initialize_gpt2_weights() {
    GPT2Weights weights;

    // Word token embeddings
    weights.wte = (float **)malloc(VOCAB_SIZE * sizeof(float *));
    for (int i = 0; i < VOCAB_SIZE; i++) {
        weights.wte[i] = (float *)malloc(EMBEDDING_SIZE * sizeof(float));
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
            weights.wte[i][j] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
        }
    }

    // Position embeddings
    weights.wpe = (float **)malloc(MAX_POSITION_EMBEDDINGS * sizeof(float *));
    for (int i = 0; i < MAX_POSITION_EMBEDDINGS; i++) {
        weights.wpe[i] = (float *)malloc(EMBEDDING_SIZE * sizeof(float));
        for (int j = 0; j < EMBEDDING_SIZE; j++) {
            weights.wpe[i][j] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
        }
    }

    // Transformer blocks: inner weight arrays now allocated and initialized
    weights.blocks = (BlockWeights *)malloc(NUM_BLOCKS * sizeof(BlockWeights));
    for (int b = 0; b < NUM_BLOCKS; b++) {
        weights.blocks[b].q_mlp         = init_linear(EMBEDDING_SIZE, EMBEDDING_SIZE);
        weights.blocks[b].k_mlp         = init_linear(EMBEDDING_SIZE, EMBEDDING_SIZE);
        weights.blocks[b].v_mlp         = init_linear(EMBEDDING_SIZE, EMBEDDING_SIZE);
        weights.blocks[b].first_block_MLP  = init_linear(EMBEDDING_SIZE, EMBEDDING_SIZE * 4);
        weights.blocks[b].second_block_MLP = init_linear(EMBEDDING_SIZE * 4, EMBEDDING_SIZE);
    }

    // Logits MLP
    weights.logits_mlp = init_linear(EMBEDDING_SIZE, VOCAB_SIZE);

    return weights;
}

// Main
int main() {
    srand(time(NULL));

    printf("Initializing GPT-2 weights...\n");
    GPT2Weights weights = initialize_gpt2_weights();

    int tokens[] = {0, 1, 2, 3, 4};
    int seqLength = sizeof(tokens) / sizeof(tokens[0]);

    printf("Running model...\n");
    float *logits = model(tokens, seqLength, weights);

    printf("First 10 logits:\n");
    for (int i = 0; i < 10; i++) {
        printf("  Logit %d: %f\n", i, logits[i]);
    }

    free(logits);
    free_gpt2_weights(weights);

    return 0;
}
