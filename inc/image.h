#ifndef IMAGE_H
#define IMAGE_H

#include "utils.h"

// 图像数据结构
typedef struct {
    unsigned char* data;  // 原始图像数据，RGB格式
    int width;           // 图像宽度
    int height;          // 图像高度
    int channels;        // 图像通道数 (通常为3表示RGB)
} Image;

// 灰度图像
typedef struct {
    float* data;      // 灰度值 (0.0-1.0)
    int width;        // 图像宽度
    int height;       // 图像高度
} GrayImage;

// CIFAR-10相关
#define CIFAR_IMAGE_SIZE 32
#define CIFAR_IMAGE_CHANNELS 3
#define CIFAR_BATCH_SIZE 10000
#define CIFAR_NUM_CLASSES 10

typedef struct {
    Image* images;          // 图像数组
    unsigned char* labels;  // 标签数组 (0-9)
    int count;              // 图像数量
} CifarDataset;

// 图像创建与释放
Image create_image(int width, int height, int channels);
void free_image(Image* img);
GrayImage create_gray_image(int width, int height);
void free_gray_image(GrayImage* img);

// 图像转换
GrayImage convert_to_gray(const Image* img);
float get_pixel_gray(const GrayImage* img, int x, int y);
void set_pixel_gray(GrayImage* img, int x, int y, float value);

// 图像操作
GrayImage compute_gradient_magnitude(const GrayImage* img);
GrayImage compute_gradient_orientation(const GrayImage* img);
GrayImage gaussian_blur(const GrayImage* img, float sigma);

// CIFAR-10操作
CifarDataset load_cifar10_batch(const char* filename);
void free_cifar_dataset(CifarDataset* dataset);
Image get_cifar_image(const CifarDataset* dataset, int index);
unsigned char get_cifar_label(const CifarDataset* dataset, int index);

// 子区域提取
Image extract_sub_image(const Image* img, int x, int y, int width, int height);
DescriptorList extract_descriptors_from_region(const Image* img, int x, int y, int width, int height);

#endif /* IMAGE_H */