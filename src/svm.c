#include "svm.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

// 初始化 SVM 模型
SVMModel* svm_create(int num_features, double C) {
    SVMModel* model = (SVMModel*)malloc(sizeof(SVMModel));
    model->weights = (double*)calloc(num_features, sizeof(double));
    model->num_features = num_features;
    model->C = C;
    return model;
}

// 训练 SVM 模型
void svm_train(SVMModel* model, double** data, int* labels, int num_samples, int max_iterations) {
    for (int iter = 0; iter < max_iterations; iter++) {
        for (int i = 0; i < num_samples; i++) {
            double prediction = 0.0;
            for (int j = 0; j < model->num_features; j++) {
                prediction += model->weights[j] * data[i][j];
            }
            if (labels[i] * prediction < 1) {
                for (int j = 0; j < model->num_features; j++) {
                    model->weights[j] += model->C * labels[i] * data[i][j];
                }
            }
        }
    }
}

// 使用 SVM 进行预测
int svm_predict(SVMModel* model, double* feature_vector) {
    double result = 0.0;
    for (int i = 0; i < model->num_features; i++) {
        result += model->weights[i] * feature_vector[i];
    }
    return result >= 0 ? 1 : -1;
}

// 释放 SVM 模型
void svm_free(SVMModel* model) {
    if (model) {
        free(model->weights);
        free(model);
    }
}