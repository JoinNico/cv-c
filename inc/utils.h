#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

// 基本数据结构
typedef struct {
    float* data;     // 特征向量数据
    int length;      // 特征向量长度
    float x, y;      // 特征点在图像中的位置
} Descriptor;

typedef struct {
    Descriptor* descriptors;  // 描述符数组
    int count;               // 描述符数量
} DescriptorList;

typedef struct {
    float** centers;    // 聚类中心
    int num_clusters;   // 聚类数量
    int dim;            // 特征维度
} Codebook;

// 内存分配函数
float* allocate_float_array(int size);
float** allocate_float_matrix(int rows, int cols);
void free_float_array(float* array);
void free_float_matrix(float** matrix, int rows);

// 描述符操作函数
Descriptor create_descriptor(int length);
void free_descriptor(Descriptor* desc);
DescriptorList create_descriptor_list(int initial_capacity);
void free_descriptor_list(DescriptorList* list);
void add_descriptor(DescriptorList* list, Descriptor desc);

// 数学工具函数
float euclidean_distance(float* v1, float* v2, int dim);
float chi_square_distance(float* v1, float* v2, int dim);
int min_int(int a, int b);
float min_float(float a, float b);

// 随机数生成
void init_random();
int random_int(int min, int max);
float random_float(float min, float max);

// 向量操作
void normalize_vector(float* vec, int length);
void vector_add(float* result, float* v1, float* v2, int length);
void vector_multiply_scalar(float* result, float* vec, float scalar, int length);
void print_vector(float* vec, int length);

// 文件操作
unsigned char* read_file(const char* filename, size_t* size);
int write_file(const char* filename, unsigned char* data, size_t size);

#endif /* UTILS_H */