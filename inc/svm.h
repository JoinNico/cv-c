#ifndef SVM_H
#define SVM_H

typedef struct {
    double* weights;  // 权重向量
    int num_features; // 特征数量
    double C;         // 惩罚参数
} SVMModel;

// 初始化 SVM 模型
SVMModel* svm_create(int num_features, double C);

// 训练 SVM 模型
void svm_train(SVMModel* model, double** data, int* labels, int num_samples, int max_iterations);

// 使用 SVM 进行预测
int svm_predict(SVMModel* model, double* feature_vector);

// 释放 SVM 模型
void svm_free(SVMModel* model);

#endif // SVM_H