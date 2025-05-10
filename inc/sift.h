#ifndef SIFT_H
#define SIFT_H

#include "image.h"

// SIFT参数
#define SIFT_DESC_SIZE 128    // SIFT描述符的维度 (4x4x8)
#define SIFT_OCTAVES 4        // 尺度空间中的八度数
#define SIFT_SCALES 5         // 每个八度中的尺度数
#define SIFT_SIGMA 1.6        // 初始高斯模糊的sigma
#define SIFT_CONTRAST_THRESH 0.03  // 对比度阈值

// 关键点检测结构
typedef struct {
    float x, y;           // 坐标
    float scale;          // 尺度
    float orientation;    // 方向 (弧度)
} KeyPoint;

typedef struct {
    KeyPoint* points;     // 关键点数组
    int count;            // 关键点数量
} KeyPointList;

// 简化版的SIFT描述符
typedef struct {
    float descriptor[SIFT_DESC_SIZE];  // 128维SIFT描述符
    float x, y;                       // 关键点在图像中的位置
} SiftDescriptor;

typedef struct {
    SiftDescriptor* descriptors;      // SIFT描述符数组
    int count;                        // 描述符数量
} SiftDescriptorList;

// 关键点检测
KeyPointList detect_keypoints(const Image* img);
void free_keypoint_list(KeyPointList* list);

// SIFT描述符计算
SiftDescriptor compute_sift_descriptor(const Image* img, KeyPoint kp);
SiftDescriptorList extract_sift_features(const Image* img);
void free_sift_descriptor_list(SiftDescriptorList* list);

// 将SIFT描述符转换为通用描述符格式
DescriptorList convert_sift_to_descriptors(const SiftDescriptorList* sift_list);

// 简化的SIFT实现（密集采样版本，用于SPM）
DescriptorList extract_dense_sift(const Image* img, int step);

#endif /* SIFT_H */