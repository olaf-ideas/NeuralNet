#pragma GCC optimize("Ofast","unroll-loops","omit-frame-pointer","inline")
#pragma GCC option("arch=native","tune=native","no-zero-upper")
#pragma GCC target("avx2","popcnt","rdrnd","bmi2")

#include <bits/stdc++.h>

#include "nnv2.h"
#include "mnist.h"

void print_data(mnist_data &data) {
    for(int i = 0; i < 28; i++) {
        for(int j = 0; j < 28; j++) {
            printf("%3d,", int(data.data[i][j]));
            
        }
        puts("");
    }
    printf("number: %u\n", data.label);
}

void load_sample(mnist_data &sample, float *input, float *target) {
    for(int i = 0; i < 28; i++) {
        for(int j = 0; j < 28; j++) {
            input[i * 28 + j] = float(sample.data[i][j]) / 255.f;
        }
    }
    
    for(int i = 0; i < 10; i++) {
        if(sample.label == i) {
            target[i] = 1.f;
        } else {
            target[i] = 0.f;
        }
    }
}

int main() {

    mnist_data *train;
    mnist_data *test;

    unsigned int train_cnt;
    int ret;
    if(ret = mnist_load("train/train-images-idx3-ubyte", "train/train-labels-idx1-ubyte", &train, &train_cnt)) {
        printf("An error occurred: %d\n", ret);
        exit(-1);
    }
    printf("data loaded, samples count: %u\n", train_cnt);
    
    unsigned int test_cnt;
    if(ret = mnist_load("test/t10k-images-idx3-ubyte", "test/t10k-labels-idx1-ubyte", &test, &test_cnt)) {
        printf("An error occurred: %d\n", ret);
        exit(-1);
    }
    printf("data loaded, samples count: %u\n", test_cnt);

    //print_data(train[0]);

    // NeuralNetwork nn({28 * 28, 64, 256, 128, 10}, {TANH, RELU, RELU, TANH}, 0.001f, 0.3f);
    NeuralNetwork nn("nn/mnist-test.nn", 0.0001f, 0.5f);

    const int BATCH_SIZE = 1000;

    for(int epoch = 0; epoch < 100; epoch++) {
        std::random_shuffle(train, train + train_cnt);
        for(int sample = 0; sample < train_cnt; sample++) {
        // for(int batch = 0; batch < 10000; batch++) {
            // int sample = rand() % train_cnt;

            static float input[28 * 28];
            static float target[10];
            static float output[10];

            load_sample(train[sample], input, target);

            nn.forward(input, output);
            nn.backprop(target);
        }
        fprintf(stderr, "training done\n");

        int correct = 0;
        for(int sample = 0; sample < test_cnt; sample++) {
            static float input[784];
            static float target[10];
            static float output[10];

            load_sample(test[sample], input, target);

            nn.forward(input, output);

            // for(int i = 0; i < 10; i++) {
            //     fprintf(stderr, "%f ", output[i]);
            // }
            // fprintf(stderr, "\n");

            int label = 0;
            float value = output[0];
            for(int i = 1; i < 10; i++) {
                if(output[i] > value) {
                    label = i;
                    value = output[i];
                }
            }

            if(test[sample].label == label) {
                correct++;
            }
        }

        printf("currectness: %f (%d/%d)\n", correct / float(test_cnt) * 100, correct, test_cnt);
    }

    nn.save("nn/mnist-test.nn");

    free(train);
    free(test);
}