#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

float rand_0_to_1() { return rand() / float(RAND_MAX); }

float  activation(float x) { return x < 0 ? 0 : x; }
float dactivation(float x) { return x < 0 ? 0 : 1; }

class NeuralNet {

private:

    float ***W; // weights
    float ***M; // momentum
    float **G;  // gradient
    float **A;  // activation

    float LEARN_RATE; // learning rate
    float MOMEN_RATE; // momentum rate

    int *L_SZ; // layers_sizes
    int L_CNT; // layers_count

public:

    void create(const int *layers_sizes, int layers_count, float learning_rate = 0.15f,
                                                           float momentum_rate = 0.5f) {
        LEARN_RATE = learning_rate;
        MOMEN_RATE = momentum_rate;

        L_CNT = layers_count;
        
        L_SZ = (int*) malloc(L_CNT * sizeof(int));
        memcpy(L_SZ, layers_sizes, L_CNT * sizeof(int));

        // WEIGHTS INITIALIZATION
        W = (float***) malloc((L_CNT - 1) * sizeof(float**));

        for(int layer = 0; layer < L_CNT - 1; layer++) {
            W[layer] = (float**) malloc((L_SZ[layer] + 1) * sizeof(float*));

            for(int node = 0; node <= L_SZ[layer]; node++) {
                W[layer][node] = (float*) malloc(L_SZ[layer + 1] * sizeof(float));
            }
        }

        // MOMENTUM INITIALIZATION
        M = (float***) malloc((L_CNT - 1) * sizeof(float**));

        for(int layer = 0; layer < L_CNT - 1; layer++) {
            M[layer] = (float**) malloc((L_SZ[layer] + 1) * sizeof(float*));
            
            for(int node = 0; node <= L_SZ[layer]; node++) {
                M[layer][node] = (float*) malloc(L_SZ[layer + 1] * sizeof(float));
            }
        }

        // GRADIENT INITIALIZATION
        G = (float**) malloc(L_CNT * sizeof(float*));
        for(int layer = 0; layer < L_CNT; layer++) {
            G[layer] = (float*) malloc((L_SZ[layer] + 1) * sizeof(float));
        }

        // ACTIVATION INITIALIZATION
        A = (float**) malloc(L_CNT * sizeof(float*));
        for(int layer = 0; layer < L_CNT; layer++) {
            A[layer] = (float*) malloc((L_SZ[layer] + 1) * sizeof(float));
        }

        reset();

    }

    NeuralNet(const std::vector<int> &layers_sizes, float learning_rate = 0.15f, 
                                                    float momentum_rate = 0.5f) {
        create(&layers_sizes[0], (int) layers_sizes.size(), learning_rate, momentum_rate);
    }

    NeuralNet(const char *file, float learning_rate = 0.15f, float momentum_rate = 0.5f) {
        FILE *nn_file = fopen(file, "rb");

        if(nn_file == NULL) {
            printf("error while loading a file\n");
        }

        int layers_count;
        fscanf(nn_file, "%d", &layers_count);

        int *layers_sizes = (int*) malloc(layers_count * sizeof(int));

        for(int layer = 0; layer < layers_count; layer++) {
            fscanf(nn_file, "%d", layers_sizes + layer);
        }

        fclose(nn_file);

        create(layers_sizes, layers_count, learning_rate, momentum_rate);

        free(layers_sizes);
    
        read(file);
    }

    void print() {
        for(int layer = 0; layer < L_CNT - 1; layer++) {
            for(int curr = 0; curr <= L_SZ[layer]; curr++) {
                for(int next = 0; next < L_SZ[layer + 1]; next++) {
                    printf("%f, ", W[layer][curr][next]);
                }
                puts("");
            }
            puts("");
        }
    }

    void save(const char *file) {
        FILE *nn_file = fopen(file, "wb");

        if(nn_file == NULL) {
            printf("error while loading a file\n");
        }

        fprintf(nn_file, "%d\n", L_CNT);
        for(int layer = 0; layer < L_CNT; layer++) {
            fprintf(nn_file, "%d ", L_SZ[layer]);
        }
        fputs("", nn_file);

        for(int layer = 0; layer < L_CNT - 1; layer++) {
            for(int curr = 0; curr < L_SZ[layer]; curr++) {
                for(int next = 0; next < L_SZ[layer + 1]; next++) {
                    fprintf(nn_file, "%f ", W[layer][curr][next]);
                }
            }
            fputs("", nn_file);
        }

        fclose(nn_file);
    }

    void read(const char *file) {
        FILE *nn_file = fopen(file, "rb");

        if(nn_file == NULL) {
            printf("error while loading a file\n");
        }

        int layers_count;
        fscanf(nn_file, "%d", &layers_count);

        if(L_CNT != layers_count) {
            printf("layers count did not match: nn(%d) file(%d)\n", L_CNT, layers_count);
        }

        for(int layer = 0; layer < layers_count; layer++) {
            int layer_size;
            fscanf(nn_file, "%d", &layer_size);

            if(layer_size != L_SZ[layer]) {
                printf("layer size did not match: nn(%d) file(%d)\n", L_SZ[layer], layer_size);
            }
        }

        for(int layer = 0; layer < L_CNT - 1; layer++) {
            for(int curr = 0; curr < L_SZ[layer]; curr++) {
                for(int next = 0; next < L_SZ[layer + 1]; next++) {
                    fscanf(nn_file, "%f ", &W[layer][curr][next]);
                }
            }
        }

        fclose(nn_file);
    }

    void reset() {
        // Weight init
        for(int layer = 0; layer < L_CNT - 1; layer++) {
            for(int curr = 0; curr <= L_SZ[layer]; curr++) {
                for(int next = 0; next < L_SZ[layer + 1]; next++) {
                    W[layer][curr][next] = rand_0_to_1() - 0.5f;
                    M[layer][curr][next] = 0;
                }
            }
        }

        // Bias init
        for(int layer = 0; layer < L_CNT - 1; layer++) {
            A[layer][L_SZ[layer]] = 1.f;
        }
    }

    void forward(const float *input, float *output) {
        for(int node = 0; node < L_SZ[0]; node++) {
            A[0][node] = input[node];
        }

        for(int layer = 0; layer < L_CNT - 1; layer++) {
            for(int next = 0; next < L_SZ[layer + 1]; next++) {
                A[layer + 1][next] = 0;
            }

            for(int curr = 0; curr <= L_SZ[layer]; curr++) {
                for(int next = 0; next < L_SZ[layer + 1]; next++) {
                    A[layer + 1][next] += A[layer][curr] * W[layer][curr][next];
                }
            }
            
            for(int next = 0; next < L_SZ[layer + 1]; next++) {
                A[layer + 1][next] = activation(A[layer + 1][next]);
            }
        }

        for(int node = 0; node < L_SZ[L_CNT - 1]; node++) {
            output[node] = A[L_CNT - 1][node];
        }
    }

    void backprop(const float *target) {
        for(int node = 0; node < L_SZ[L_CNT - 1]; node++) {
            G[L_CNT - 1][node] = (target[node] - A[L_CNT - 1][node]) 
                                * dactivation(A[L_CNT - 1][node]);
        }

        for(int layer = L_CNT - 2; layer > 0; layer--) {
            for(int curr = 0; curr <= L_SZ[layer]; curr++) {
                float sum = 0;
                for(int next = 0; next < L_SZ[layer + 1]; next++) {
                    sum += W[layer][curr][next] * G[layer + 1][next];
                }
                G[layer][curr] = sum * dactivation(A[layer][curr]);
            }
        }

        for(int layer = L_CNT - 2; layer >= 0; layer--) {
            for(int curr = 0; curr <= L_SZ[layer]; curr++) {
                for(int next = 0; next < L_SZ[layer + 1]; next++) {
                    float &delta = M[layer][curr][next];
                    delta = LEARN_RATE * A[layer][curr] * G[layer + 1][next] 
                          + MOMEN_RATE * delta;

                    W[layer][curr][next] += delta;
                }
            }
        }
    }

    ~NeuralNet() {

        // prosze się nie wywalic
        for(int layer = 0; layer < L_CNT - 1; layer++) {
            for(int node = 0; node <= L_SZ[layer]; node++) {
                free(W[layer][node]);
            }
            free(W[layer]);
        }
        free(W);

        for(int layer = 0; layer < L_CNT - 1; layer++) {
            for(int node = 0; node <= L_SZ[layer]; node++)
                free(M[layer][node]);
            free(M[layer]);
        }
        free(M);

        
        for(int layer = 0; layer < L_CNT; layer++)
            free(G[layer]);
        free(G);

        for(int layer = 0; layer < L_CNT; layer++)
            free(A[layer]);
        free(A);

        free(L_SZ);
    }
};

#endif // NN_H