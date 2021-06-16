#ifndef NNv2_H
#define NNv2_H

#pragma GCC optimize("Ofast","unroll-loops","omit-frame-pointer","inline")
#pragma GCC option("arch=native","tune=native","no-zero-upper")
#pragma GCC target("avx2","popcnt","rdrnd","bmi2")

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>

enum ActivationType : int {
    RELU = 0,
    TANH = 1,
    SIGMOID = 2,
    LINEAR = 3,
};

inline unsigned int fastrand() { 
    static unsigned int g_seed = 2137;
    g_seed = (214013 * g_seed + 2531011); 
    return (g_seed >> 16) & 0x7FFF; 
}

float activation(float x, ActivationType type) {
    switch(type) {
        case RELU:      return x > 0 ? x : 0;
        case TANH:      return tanh(x);
        case SIGMOID:   return 1 / (1 - exp(x));
        case LINEAR:    return x;
    }

    fprintf(stderr, "error in activation functions\n");
    fprintf(stderr, "x = %f type = %d\n", x, (int) type);
    exit(-1);
}

float derivative(float x, ActivationType type) {
    switch(type) {
        case RELU:      return x > 0 ? 1 : 0;
        case TANH:      return 1 - x * x;
        case SIGMOID:   return x * (1 - x);
        case LINEAR:    return 1;
    }

    fprintf(stderr, "error in derivative functions\n");
    fprintf(stderr, "x = %f type = %d\n", x, (int) type);
    exit(-1);
}

class NeuralNetwork {

public:

    void create(const int *layers_sizes, int layers_count, const ActivationType *activation) {

        m_layers_count = layers_count;
        m_layers_sizes = (int*) malloc(m_layers_count * sizeof(int));
        memcpy(m_layers_sizes, layers_sizes, m_layers_count * sizeof(int));
    
        m_weights = (float***) malloc((m_layers_count - 1) * sizeof(float**));

        for(int layer = 0; layer < m_layers_count - 1; layer++) {
            m_weights[layer] = (float**) malloc((m_layers_sizes[layer] + 1) * sizeof(float*));
            for(int node = 0; node <= m_layers_sizes[layer]; node++) {
                m_weights[layer][node] = (float*) malloc(m_layers_sizes[layer + 1] * sizeof(float));
            }
        }

        m_momentum = (float***) malloc((m_layers_count - 1) * sizeof(float**));
        for(int layer = 0; layer < m_layers_count - 1; layer++) {
            m_momentum[layer] = (float**) malloc((m_layers_sizes[layer] + 1) * sizeof(float*));
            for(int node = 0; node <= m_layers_sizes[layer]; node++) {
                m_momentum[layer][node] = (float*) malloc(m_layers_sizes[layer + 1] * sizeof(float));
            }
        }

        m_value = (float**) malloc(m_layers_count * sizeof(float*));
        for(int layer = 0; layer < m_layers_count; layer++) {
            m_value[layer] = (float*) malloc((m_layers_sizes[layer] + 1) * sizeof(float));
        }

        m_gradient = (float**) malloc(m_layers_count * sizeof(float*));
        for(int layer = 0; layer < m_layers_count; layer++) {
            m_gradient[layer] = (float*) malloc((m_layers_sizes[layer] + 1) * sizeof(float));
        }

        m_activation = (ActivationType*) malloc((m_layers_count - 1) * sizeof(ActivationType));
        memcpy(m_activation, activation, (m_layers_count - 1) * sizeof(ActivationType));
    }

    void reset() {
        for(int layer = 0; layer < m_layers_count - 1; layer++) {
            for(int curr = 0; curr <= m_layers_sizes[layer]; curr++) {
                float bound = sqrt(6.f / (m_layers_sizes[layer] + m_layers_sizes[layer + 1]));
                for(int next = 0; next < m_layers_sizes[layer + 1]; next++) {
                    m_weights[layer][curr][next] = (fastrand() / 32768.f - 0.5f) * bound;
                    m_momentum[layer][curr][next] = 0.f;
                }
            }
        }

        for(int layer = 0; layer < m_layers_count - 1; layer++) {
            m_value[layer][m_layers_sizes[layer]] = 1.f;
        }
    }

    void save(const char *filename) {
        FILE *nn_file = fopen(filename, "wb");

        if(nn_file == NULL) {
            fprintf(stderr, "error while loading a file '%s'\n", filename);
            exit(-1);
        }

        fprintf(nn_file, "%d\n", m_layers_count);
        for(int layer = 0; layer < m_layers_count; layer++) {
            fprintf(nn_file, "%d ", m_layers_sizes[layer]);
        }
        fprintf(nn_file, "\n");

        for(int layer = 0; layer < m_layers_count - 1; layer++) {
            fprintf(nn_file, "%d ", int(m_activation[layer]));
        }
        fprintf(nn_file, "\n");

        for(int layer = 0; layer < m_layers_count - 1; layer++) {
            for(int curr = 0; curr <= m_layers_sizes[layer]; curr++) {
                for(int next = 0; next < m_layers_sizes[layer + 1]; next++) {
                    fprintf(nn_file, "%f ", m_weights[layer][curr][next]);
                }
            }
            fputs("", nn_file);
        }

        fclose(nn_file);
    }

    void load(const char *filename) {
        FILE *nn_file = fopen(filename, "rb");

        if(nn_file == NULL) {
            fprintf(stderr, "error while loading a file '%s'\n", filename);
            exit(-1);
        }

        int layers_count;
        fscanf(nn_file, "%d", &layers_count);

        int* layers_sizes = (int*) malloc(layers_count * sizeof(int));
        for(int layer = 0; layer < layers_count; layer++) {
            fscanf(nn_file, "%d", &layers_sizes[layer]);
        }

        ActivationType *activation = (ActivationType*) malloc((layers_count - 1) * sizeof(ActivationType));
        for(int layer = 0; layer < layers_count - 1; layer++) {
            int type;
            fscanf(nn_file, "%d", &type);
            activation[layer] = ActivationType(type);
        }

        create(layers_sizes, layers_count, activation);

        for(int layer = 0; layer < layers_count - 1; layer++) {
            for(int curr = 0; curr <= layers_sizes[layer]; curr++) {
                for(int next = 0; next < layers_sizes[layer + 1]; next++) {
                    fscanf(nn_file, "%f", &m_weights[layer][curr][next]);
                }
            }
        }

        free(layers_sizes);
        free(activation);

        fclose(nn_file);
    }

    void load(std::stringstream &ss) {

        int layers_count;
        ss >> layers_count;

        int* layers_sizes = (int*) malloc(layers_count * sizeof(int));
        for(int layer = 0; layer < layers_count; layer++) {
            ss >> layers_sizes[layer];
        }

        ActivationType *activation = (ActivationType*) malloc((layers_count - 1) * sizeof(ActivationType));
        for(int layer = 0; layer < layers_count - 1; layer++) {
            int type;
            ss >> type;
            activation[layer] = ActivationType(type);
        }

        create(layers_sizes, layers_count, activation);

        for(int layer = 0; layer < layers_count - 1; layer++) {
            for(int curr = 0; curr <= layers_sizes[layer]; curr++) {
                for(int next = 0; next < layers_sizes[layer + 1]; next++) {
                    ss >> m_weights[layer][curr][next];
                }
            }
        }

        free(layers_sizes);
        free(activation);
    }

    void forward(const float *input, float *output) {
        for(int node = 0; node < m_layers_sizes[0]; node++) {
            m_value[0][node] = input[node];
        }

        for(int layer = 0; layer < m_layers_count - 1; layer++) {
            for(int next = 0; next < m_layers_sizes[layer + 1]; next++) {
                m_value[layer + 1][next] = 0;
            }

            for(int curr = 0; curr <= m_layers_sizes[layer]; curr++) {
                for(int next = 0; next < m_layers_sizes[layer + 1]; next++) {
                    m_value[layer + 1][next] += m_value[layer][curr] * m_weights[layer][curr][next];
                }
            }

            for(int next = 0; next < m_layers_sizes[layer + 1]; next++) {
                m_value[layer + 1][next] = activation(m_value[layer + 1][next], m_activation[layer]);
            }
        }

        for(int node = 0; node < m_layers_sizes[m_layers_count - 1]; node++) {
            output[node] = m_value[m_layers_count - 1][node];
        }
    }

    void forward(const std::vector<float> &input, std::vector<float> &output) {
        output.resize(m_layers_sizes[m_layers_count - 1]);
        forward(&input[0], &output[0]);
    }

    void backprop(const float *target) {
        for(int node = 0; node < m_layers_sizes[m_layers_count - 1]; node++) {
            m_gradient[m_layers_count - 1][node] = (target[node] - m_value[m_layers_count - 1][node]) *
                                                derivative(m_value[m_layers_count - 1][node], m_activation[m_layers_count - 2]);
        }

        for(int layer = m_layers_count - 2; layer > 0; layer--) {
            for(int curr = 0; curr <= m_layers_sizes[layer]; curr++) {
                float sum = 0;
                for(int next = 0; next < m_layers_sizes[layer + 1]; next++) {
                    sum += m_weights[layer][curr][next] * m_gradient[layer + 1][next];
                }
                m_gradient[layer][curr] = sum * derivative(m_value[layer][curr], m_activation[layer - 1]);
            }
        }

        for(int layer = m_layers_count - 2; layer >= 0; layer--) {
            for(int curr = 0; curr <= m_layers_sizes[layer]; curr++) {
                for(int next = 0; next < m_layers_sizes[layer + 1]; next++) {
                    float &delta = m_momentum[layer][curr][next];
                    delta = m_learning_rate * m_value[layer][curr] * m_gradient[layer + 1][next] +
                            m_momentum_rate * delta;
                    m_weights[layer][curr][next] += delta;
                }
            }
        }
    }

    void backprop(const std::vector<float> &target) {
        backprop(&target[0]);
    }

    NeuralNetwork(const std::vector<int> &layers_sizes, const std::vector<ActivationType> &activation,
                  float learning_rate = 0.15f, float momentum_rate = 0.5f) :
        m_learning_rate(learning_rate), m_momentum_rate(momentum_rate) {
        create(&layers_sizes[0], (int) layers_sizes.size(), &activation[0]);
        reset();
    }

    NeuralNetwork(const char *filename,
                  float learning_rate = 0.15f, float momentum_rate = 0.5f) : 
        m_learning_rate(learning_rate), m_momentum_rate(momentum_rate) {
        load(filename);
    }

    NeuralNetwork(std::stringstream &ss) {
        load(ss);
    }

    ~NeuralNetwork() {
        free(m_activation);
        for(int layer = 0; layer < m_layers_count; layer++) {
            free(m_gradient[layer]);
        }
        free(m_gradient);
        for(int layer = 0; layer < m_layers_count; layer++) {
            free(m_value[layer]);
        }
        free(m_value);
        for(int layer = 0; layer < m_layers_count - 1; layer++) {
            for(int node = 0; node <= m_layers_sizes[layer]; node++) {
                free(m_momentum[layer][node]);
            }
            free(m_momentum[layer]);
        }
        free(m_momentum);
        for(int layer = 0; layer < m_layers_count - 1; layer++) {
            for(int node = 0; node <= m_layers_sizes[layer]; node++) {
                free(m_weights[layer][node]);
            }
            free(m_weights[layer]);
        }
        free(m_weights);
        free(m_layers_sizes);
    }

    void print() {
        for(int layer = 0; layer < m_layers_count - 1; layer++) {
            for(int curr = 0; curr <= m_layers_sizes[layer]; curr++) {
                for(int next = 0; next < m_layers_sizes[layer + 1]; next++) {
                    fprintf(stderr, "%f, ", m_weights[layer][curr][next]);
                }
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "\n");
        }
    }

private:

    int  m_layers_count;
    int *m_layers_sizes;

    float ***m_weights;
    float ***m_momentum;

    float **m_value;
    float **m_gradient;

    ActivationType *m_activation;
    
    float m_learning_rate;
    float m_momentum_rate;

};

#endif