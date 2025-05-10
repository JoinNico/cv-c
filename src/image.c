#include "image.h"
#include "utils.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// 图像创建与释放
Image create_image(int width, int height, int channels) {
    Image img;
    img.width = width;
    img.height = height;
    img.channels = channels;
    img.data = (unsigned char*)malloc(width * height * channels * sizeof(unsigned char));

    if (!img.data) {
        fprintf(stderr, "Error: Memory allocation failed for image\n");
        exit(EXIT_FAILURE);
    }

    memset(img.data, 0, width * height * channels * sizeof(unsigned char));
    return img;
}

void free_image(Image* img) {
    if (img && img->data) {
        free(img->data);
        img->data = NULL;
        img->width = 0;
        img->height = 0;
        img->channels = 0;
    }
}

GrayImage create_gray_image(int width, int height) {
    GrayImage img;
    img.width = width;
    img.height = height;
    img.data = (float*)malloc(width * height * sizeof(float));

    if (!img.data) {
        fprintf(stderr, "Error: Memory allocation failed for gray image\n");
        exit(EXIT_FAILURE);
    }

    memset(img.data, 0, width * height * sizeof(float));
    return img;
}

void free_gray_image(GrayImage* img) {
    if (img && img->data) {
        free(img->data);
        img->data = NULL;
        img->width = 0;
        img->height = 0;
    }
}

// 图像转换
GrayImage convert_to_gray(const Image* img) {
    GrayImage gray = create_gray_image(img->width, img->height);

    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            int idx = (y * img->width + x) * img->channels;

            // 使用BT.709加权平均将RGB转换为灰度
            float r = img->data[idx] / 255.0f;
            float g = img->data[idx + 1] / 255.0f;
            float b = img->data[idx + 2] / 255.0f;

            float gray_value = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            gray.data[y * img->width + x] = gray_value;
        }
    }

    return gray;
}

float get_pixel_gray(const GrayImage* img, int x, int y) {
    // 边界检查
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img->width) x = img->width - 1;
    if (y >= img->height) y = img->height - 1;

    return img->data[y * img->width + x];
}

void set_pixel_gray(GrayImage* img, int x, int y, float value) {
    // 边界检查
    if (x < 0 || y < 0 || x >= img->width || y >= img->height) {
        return;
    }

    img->data[y * img->width + x] = value;
}

// 图像操作
GrayImage compute_gradient_magnitude(const GrayImage* img) {
    GrayImage magnitude = create_gray_image(img->width, img->height);

    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            // 简单的Sobel算子计算梯度
            float gx = get_pixel_gray(img, x+1, y) - get_pixel_gray(img, x-1, y);
            float gy = get_pixel_gray(img, x, y+1) - get_pixel_gray(img, x, y-1);

            // 计算梯度幅值
            float mag = sqrtf(gx*gx + gy*gy);
            set_pixel_gray(&magnitude, x, y, mag);
        }
    }

    return magnitude;
}

GrayImage compute_gradient_orientation(const GrayImage* img) {
    GrayImage orientation = create_gray_image(img->width, img->height);

    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            // 简单的Sobel算子计算梯度
            float gx = get_pixel_gray(img, x+1, y) - get_pixel_gray(img, x-1, y);
            float gy = get_pixel_gray(img, x, y+1) - get_pixel_gray(img, x, y-1);

            // 计算梯度方向 (弧度，范围 [0, 2π))
            float angle = atan2f(gy, gx);
            if (angle < 0) angle += 2.0f * M_PI;

            set_pixel_gray(&orientation, x, y, angle);
        }
    }

    return orientation;
}

GrayImage gaussian_blur(const GrayImage* img, float sigma) {
    // 计算高斯核大小
    int kernel_size = (int)(6.0f * sigma + 1.0f);
    if (kernel_size % 2 == 0) kernel_size++;  // 确保kernel_size是奇数

    int half_size = kernel_size / 2;

    // 创建高斯核
    float* kernel = (float*)malloc(kernel_size * sizeof(float));
    float sum = 0.0f;

    for (int i = 0; i < kernel_size; i++) {
        int x = i - half_size;
        kernel[i] = expf(-(x*x) / (2.0f * sigma * sigma));
        sum += kernel[i];
    }

    // 归一化高斯核
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }

    // 水平方向高斯模糊
    GrayImage temp = create_gray_image(img->width, img->height);
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            float sum = 0.0f;
            for (int i = -half_size; i <= half_size; i++) {
                int xi = x + i;
                if (xi < 0) xi = 0;
                if (xi >= img->width) xi = img->width - 1;

                sum += get_pixel_gray(img, xi, y) * kernel[i + half_size];
            }
            set_pixel_gray(&temp, x, y, sum);
        }
    }

    // 垂直方向高斯模糊
    GrayImage blurred = create_gray_image(img->width, img->height);
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            float sum = 0.0f;
            for (int i = -half_size; i <= half_size; i++) {
                int yi = y + i;
                if (yi < 0) yi = 0;
                if (yi >= img->height) yi = img->height - 1;

                sum += get_pixel_gray(&temp, x, yi) * kernel[i + half_size];
            }
            set_pixel_gray(&blurred, x, y, sum);
        }
    }

    free_gray_image(&temp);
    free(kernel);

    return blurred;
}

// CIFAR-10操作
CifarDataset load_cifar10_batch(const char* filename) {
    CifarDataset dataset;
    dataset.count = 0;
    dataset.images = NULL;
    dataset.labels = NULL;

    size_t size;
    unsigned char* data = read_file(filename, &size);
    if (!data) {
        fprintf(stderr, "Error: Could not read CIFAR-10 batch file\n");
        return dataset;
    }

    // CIFAR-10文件格式：每个样本有1个字节的标签和3072个字节的图像数据(32x32x3)
    int num_samples = size / (1 + CIFAR_IMAGE_SIZE * CIFAR_IMAGE_SIZE * CIFAR_IMAGE_CHANNELS);

    dataset.count = num_samples;
    dataset.images = (Image*)malloc(num_samples * sizeof(Image));
    dataset.labels = (unsigned char*)malloc(num_samples * sizeof(unsigned char));

    if (!dataset.images || !dataset.labels) {
        fprintf(stderr, "Error: Memory allocation failed for CIFAR dataset\n");
        free(data);
        if (dataset.images) free(dataset.images);
        if (dataset.labels) free(dataset.labels);
        dataset.count = 0;
        dataset.images = NULL;
        dataset.labels = NULL;
        return dataset;
    }

    for (int i = 0; i < num_samples; i++) {
        int offset = i * (1 + CIFAR_IMAGE_SIZE * CIFAR_IMAGE_SIZE * CIFAR_IMAGE_CHANNELS);

        // 读取标签
        dataset.labels[i] = data[offset];

        // 读取图像数据
        dataset.images[i] = create_image(CIFAR_IMAGE_SIZE, CIFAR_IMAGE_SIZE, CIFAR_IMAGE_CHANNELS);

        // CIFAR-10中的数据是按RGB分通道存储的
        // 先是所有像素的R通道，然后是G通道，最后是B通道
        for (int c = 0; c < CIFAR_IMAGE_CHANNELS; c++) {
            for (int y = 0; y < CIFAR_IMAGE_SIZE; y++) {
                for (int x = 0; x < CIFAR_IMAGE_SIZE; x++) {
                    int img_offset = offset + 1 + c * CIFAR_IMAGE_SIZE * CIFAR_IMAGE_SIZE + y * CIFAR_IMAGE_SIZE + x;
                    int pixel_offset = (y * CIFAR_IMAGE_SIZE + x) * CIFAR_IMAGE_CHANNELS + c;
                    dataset.images[i].data[pixel_offset] = data[img_offset];
                }
            }
        }
    }

    free(data);
    return dataset;
}

void free_cifar_dataset(CifarDataset* dataset) {
    if (dataset) {
        if (dataset->images) {
            for (int i = 0; i < dataset->count; i++) {
                free_image(&dataset->images[i]);
            }
            free(dataset->images);
            dataset->images = NULL;
        }

        if (dataset->labels) {
            free(dataset->labels);
            dataset->labels = NULL;
        }

        dataset->count = 0;
    }
}

Image get_cifar_image(const CifarDataset* dataset, int index) {
    if (index < 0 || index >= dataset->count) {
        fprintf(stderr, "Error: Invalid index for CIFAR dataset\n");
        return create_image(0, 0, 0);
    }

    return dataset->images[index];
}

unsigned char get_cifar_label(const CifarDataset* dataset, int index) {
    if (index < 0 || index >= dataset->count) {
        fprintf(stderr, "Error: Invalid index for CIFAR dataset\n");
        return 0;
    }

    return dataset->labels[index];
}

// 子区域提取
Image extract_sub_image(const Image* img, int x, int y, int width, int height) {
    // 确保子区域在原图像范围内
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x + width > img->width) width = img->width - x;
    if (y + height > img->height) height = img->height - y;

    Image sub = create_image(width, height, img->channels);

    for (int sy = 0; sy < height; sy++) {
        for (int sx = 0; sx < width; sx++) {
            int orig_idx = ((y + sy) * img->width + (x + sx)) * img->channels;
            int sub_idx = (sy * width + sx) * img->channels;

            for (int c = 0; c < img->channels; c++) {
                sub.data[sub_idx + c] = img->data[orig_idx + c];
            }
        }
    }

    return sub;
}