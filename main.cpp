#include <bits/stdc++.h>

#include "nn.h"
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

    NeuralNet nn("nn/mnist-zoga4.nn", 0.005f, 0.1f);

    for(int epoch = 0; epoch < 10; epoch++) {
        int input_done = 0;

        std::random_shuffle(train, train + train_cnt);

        for(int batch = 0; batch < train_cnt; batch += 100) {
            for(int rep = 0; rep < 1000; rep++) {
                int sample = rand() % 100 + batch;

                static float input[784];
                static float target[10];
                static float output[10];

                load_sample(train[sample], input, target);

                nn.forward(input, output);
                nn.backprop(target);
            }
        }

        int correct = 0;
        for(int sample = 0; sample < test_cnt; sample++) {
            static float input[784];
            static float target[10];
            static float output[10];

            load_sample(test[sample], input, target);

            nn.forward(input, output);

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

        printf("currectness: %f %\n", correct / float(test_cnt) * 100);
    }

    nn.save("nn/mnist-zoga5.nn");

    free(train);
    free(test);
}