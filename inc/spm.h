#ifndef SPM_H
#define SPM_H

#include "kmeans.h"
#include "image.h"

// SPM级别定义
#define SPM_LEVEL_0 0  // 1x1网格
#define SPM_LEVEL_1 1  // 2x2网格
#define SPM_LEVEL_2 2  // 4x4网格

// SPM相关数据结构
typedef struct {
    float* histogram;   // SPM直方图
    int length;         // 直方图长度
} SpmHistogram;

// SPM功能
// 将图像分成金字塔层次的网格并提取描述符
DescriptorList* extract_pyramid_descriptors(const Image* img, int level);

// 从一个区域提取描述符
DescriptorList extract_region_descriptors(const Image* img, int x, int y, int width, int height);

// 构建空间金字塔直方图
SpmHistogram build_spatial_pyramid(const Image* img, Codebook* codebook, int level);

// 释放SPM直方图
void free_spm_histogram(SpmHistogram* hist);

// 计算两个SPM直方图之间的相似度（使用chi-square距离）
float compute_spm_similarity(const SpmHistogram* hist1, const SpmHistogram* hist2);

// SPM训练和测试函数
// 从图像构建码本
Codebook build_codebook_from_images(Image* images, int num_images, int voc_size);

// 计算一组图像的SPM特征
SpmHistogram* compute_spm_features(Image* images, int num_images, Codebook* codebook, int level);

#endif /* SPM_H */