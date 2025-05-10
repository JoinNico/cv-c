#ifndef KMEANS_H
#define KMEANS_H

#include "utils.h"

// K-means聚类结果
typedef struct {
    float** centers;    // 聚类中心
    int* assignments;   // 每个点的簇分配
    int num_clusters;   // 簇的数量
    int dim;            // 特征维度
    int num_points;     // 数据点数量
} KMeansResult;

// 执行K-means聚类
KMeansResult kmeans_cluster(float** data, int num_points, int dim, int num_clusters, int max_iter);

// 释放K-means结果
void free_kmeans_result(KMeansResult* result);

// 从描述符列表构建码本
Codebook build_codebook(DescriptorList* descriptors, int num_clusters);

// 释放码本
void free_codebook(Codebook* codebook);

// 计算描述符到码本中心的最近距离
int find_nearest_center(float* desc, Codebook* codebook);

// 将描述符量化为直方图
float* quantize_descriptors(DescriptorList* descriptors, Codebook* codebook);

#endif /* KMEANS_H */