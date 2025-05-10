#include "spm.h"
#include "kmeans.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// 将图像分成金字塔层次的网格并提取描述符
DescriptorList* extract_pyramid_descriptors(const Image* img, int level) {
    // 根据 SPM 的 level，将图像分成网格
    int grid_size = 1 << level; // 2^level
    DescriptorList* descriptors = (DescriptorList*)malloc(grid_size * grid_size * sizeof(DescriptorList));

    int cell_width = img->width / grid_size;
    int cell_height = img->height / grid_size;

    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            descriptors[i * grid_size + j] = extract_region_descriptors(
                    img, j * cell_width, i * cell_height, cell_width, cell_height);
        }
    }

    return descriptors;
}

// 从一个区域提取描述符
DescriptorList extract_region_descriptors(const Image* img, int x, int y, int width, int height) {
    DescriptorList descriptors;
    // 简单的占位符实现
    descriptors.count = (width * height) / 100; // 假设每 100 个像素提取一个描述符
    descriptors.descriptors = (Descriptor*)malloc(descriptors.count * sizeof(Descriptor));

    for (int i = 0; i < descriptors.count; i++) {
        // 模拟填充描述符
        descriptors.descriptors[i].data = (float*)malloc(128 * sizeof(float)); // 假设每个描述符有 128 维
        memset(descriptors.descriptors[i].data, 0, 128 * sizeof(float));
    }

    return descriptors;
}

// 构建空间金字塔直方图
SpmHistogram build_spatial_pyramid(const Image* img, Codebook* codebook, int level) {
    SpmHistogram hist;
    int total_bins = codebook->size * ((1 << (2 * (level + 1))) - 1) / 3; // 计算所有金字塔层的总 bin 数
    hist.histogram = (float*)calloc(total_bins, sizeof(float));
    hist.length = total_bins;

    DescriptorList* descriptors = extract_pyramid_descriptors(img, level);

    // 根据描述符更新直方图
    for (int i = 0; i < (1 << level) * (1 << level); i++) {
        for (int j = 0; j < descriptors[i].count; j++) {
            int bin = kmeans_assign_to_cluster(codebook, descriptors[i].descriptors[j].data);
            hist.histogram[bin]++;
        }
        free(descriptors[i].descriptors);
    }
    free(descriptors);

    return hist;
}

// 释放SPM直方图
void free_spm_histogram(SpmHistogram* hist) {
    if (hist && hist->histogram) {
        free(hist->histogram);
        hist->histogram = NULL;
        hist->length = 0;
    }
}

// 计算两个SPM直方图之间的相似度（使用chi-square距离）
float compute_spm_similarity(const SpmHistogram* hist1, const SpmHistogram* hist2) {
    if (!hist1 || !hist2 || hist1->length != hist2->length) {
        return -1.0f; // 错误处理
    }

    float similarity = 0.0f;
    for (int i = 0; i < hist1->length; i++) {
        if (hist1->histogram[i] + hist2->histogram[i] > 0) {
            float diff = hist1->histogram[i] - hist2->histogram[i];
            similarity += (diff * diff) / (hist1->histogram[i] + hist2->histogram[i]);
        }
    }

    return 1.0f - similarity; // 返回相似度
}

// 从图像构建码本
Codebook build_codebook_from_images(Image* images, int num_images, int voc_size) {
    Codebook codebook;
    codebook.size = voc_size;
    codebook.centroids = (float**)malloc(voc_size * sizeof(float*));

    // 模拟 KMeans 训练
    for (int i = 0; i < voc_size; i++) {
        codebook.centroids[i] = (float*)malloc(128 * sizeof(float)); // 假设每个特征有 128 维
        memset(codebook.centroids[i], 0, 128 * sizeof(float));
    }

    return codebook;
}

// 计算一组图像的SPM特征
SpmHistogram* compute_spm_features(Image* images, int num_images, Codebook* codebook, int level) {
    SpmHistogram* histograms = (SpmHistogram*)malloc(num_images * sizeof(SpmHistogram));
    for (int i = 0; i < num_images; i++) {
        histograms[i] = build_spatial_pyramid(&images[i], codebook, level);
    }
    return histograms;
}