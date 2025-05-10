#include "sift.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// 添加关键点到列表
static void add_keypoint(KeyPointList* list, float x, float y, float scale, float orientation) {
    KeyPoint* new_points = (KeyPoint*)realloc(list->points, (list->count + 1) * sizeof(KeyPoint));

    if (!new_points) {
        fprintf(stderr, "Error: Memory reallocation failed for keypoint list\n");
        return;
    }

    list->points = new_points;
    list->points[list->count].x = x;
    list->points[list->count].y = y;
    list->points[list->count].scale = scale;
    list->points[list->count].orientation = orientation;
    list->count++;
}

// 增加SIFT描述符到列表
static void add_sift_descriptor(SiftDescriptorList* list, SiftDescriptor desc) {
    SiftDescriptor* new_descs = (SiftDescriptor*)realloc(list->descriptors,
                                                         (list->count + 1) * sizeof(SiftDescriptor));

    if (!new_descs) {
        fprintf(stderr, "Error: Memory reallocation failed for SIFT descriptor list\n");
        return;
    }

    list->descriptors = new_descs;
    list->descriptors[list->count] = desc;
    list->count++;
}

// 简化版关键点检测
// 注意：这是SIFT算法的一个非常简化版本，实际中应该使用完整的DoG+极值检测
KeyPointList detect_keypoints(const Image* img) {
    KeyPointList list = {NULL, 0};

    // 将图像转换为灰度
    GrayImage gray = convert_to_gray(img);

    // 计算梯度幅值和方向
    GrayImage magnitude = compute_gradient_magnitude(&gray);
    GrayImage orientation = compute_gradient_orientation(&gray);

    // 简化起见，我们以规则网格上的点作为关键点
    // 实际的SIFT应该寻找DoG空间中的局部极值点
    int step = 4;  // 采样步长
    for (int y = step; y < img->height - step; y += step) {
        for (int x = step; x < img->width - step; x += step) {
            float grad_mag = get_pixel_gray(&magnitude, x, y);

            // 只保留梯度幅值足够大的点
            if (grad_mag > SIFT_CONTRAST_THRESH) {
                float angle = get_pixel_gray(&orientation, x, y);
                add_keypoint(&list, (float)x, (float)y, SIFT_SIGMA, angle);
            }
        }
    }

    free_gray_image(&gray);
    free_gray_image(&magnitude);
    free_gray_image(&orientation);

    return list;
}

void free_keypoint_list(KeyPointList* list) {
    if (list && list->points) {
        free(list->points);
        list->points = NULL;
        list->count = 0;
    }
}

// 计算SIFT描述符
// 注意：这是一个简化版本
SiftDescriptor compute_sift_descriptor(const Image* img, KeyPoint kp) {
    SiftDescriptor desc;
    memset(desc.descriptor, 0, SIFT_DESC_SIZE * sizeof(float));
    desc.x = kp.x;
    desc.y = kp.y;

    // 将图像转换为灰度
    GrayImage gray = convert_to_gray(img);

    // 计算梯度幅值和方向
    GrayImage magnitude = compute_gradient_magnitude(&gray);
    GrayImage orientation = compute_gradient_orientation(&gray);

    // SIFT描述符是4x4的网格，每个单元有8个方向直方图
    // 假设每个单元的大小为4x4像素
    int grid_size = 4;
    int cell_size = 4;
    int bins = 8;  // 8个方向

    // 以关键点为中心提取描述符
    int half_width = grid_size * cell_size / 2;

    for (int grid_y = 0; grid_y < grid_size; grid_y++) {
        for (int grid_x = 0; grid_x < grid_size; grid_x++) {
            // 计算当前单元格的起始位置
            int cell_start_x = (int)kp.x - half_width + grid_x * cell_size;
            int cell_start_y = (int)kp.y - half_width + grid_y * cell_size;

            // 遍历单元格内的每个像素
            for (int cell_y = 0; cell_y < cell_size; cell_y++) {
                for (int cell_x = 0; cell_x < cell_size; cell_x++) {
                    int x = cell_start_x + cell_x;
                    int y = cell_start_y + cell_y;

                    // 确保在图像边界内
                    if (x >= 0 && x < img->width && y >= 0 && y < img->height) {
                        float mag = get_pixel_gray(&magnitude, x, y);
                        float angle = get_pixel_gray(&orientation, x, y);

                        // 将角度归一化到0-2PI
                        if (angle < 0) angle += 2.0f * M_PI;

                        // 计算角度所在的bin
                        int bin = (int)(bins * angle / (2.0f * M_PI)) % bins;

                        // 计算对应的描述符索引
                        int descriptor_idx = (grid_y * grid_size + grid_x) * bins + bin;

                        // 累加梯度幅值
                        desc.descriptor[descriptor_idx] += mag;
                    }
                }
            }
        }
    }

    // 归一化描述符
    float sum = 0.0f;
    for (int i = 0; i < SIFT_DESC_SIZE; i++) {
        sum += desc.descriptor[i] * desc.descriptor[i];
    }

    if (sum > 0) {
        float norm = 1.0f / sqrtf(sum);
        for (int i = 0; i < SIFT_DESC_SIZE; i++) {
            desc.descriptor[i] *= norm;
        }
    }

    free_gray_image(&gray);
    free_gray_image(&magnitude);
    free_gray_image(&orientation);

    return desc;
}

// 从整个图像中提取SIFT特征
SiftDescriptorList extract_sift_features(const Image* img) {
    SiftDescriptorList list = {NULL, 0};

    // 检测关键点
    KeyPointList keypoints = detect_keypoints(img);

    // 为每个关键点计算SIFT描述符
    for (int i = 0; i < keypoints.count; i++) {
        SiftDescriptor desc = compute_sift_descriptor(img, keypoints.points[i]);
        add_sift_descriptor(&list, desc);
    }

    free_keypoint_list(&keypoints);

    return list;
}

void free_sift_descriptor_list(SiftDescriptorList* list) {
    if (list && list->descriptors) {
        free(list->descriptors);
        list->descriptors = NULL;
        list->count = 0;
    }
}

// 将SIFT描述符列表转换为通用描述符列表
DescriptorList convert_sift_to_descriptors(const SiftDescriptorList* sift_list) {
    DescriptorList list = create_descriptor_list(sift_list->count);

    for (int i = 0; i < sift_list->count; i++) {
        Descriptor desc = create_descriptor(SIFT_DESC_SIZE);

        // 复制描述符数据
        memcpy(desc.data, sift_list->descriptors[i].descriptor, SIFT_DESC_SIZE * sizeof(float));

        // 复制坐标
        desc.x = sift_list->descriptors[i].x;
        desc.y = sift_list->descriptors[i].y;

        add_descriptor(&list, desc);
    }

    return list;
}

// 提取密集SIFT特征 (在规则网格上提取)
DescriptorList extract_dense_sift(const Image* img, int step) {
    DescriptorList list = create_descriptor_list(0);

    // 将图像转换为灰度
    GrayImage gray = convert_to_gray(img);

    // 计算梯度幅值和方向
    GrayImage magnitude = compute_gradient_magnitude(&gray);
    GrayImage orientation = compute_gradient_orientation(&gray);

    // 在规则网格上提取SIFT特征
    for (int y = step; y < img->height - step; y += step) {
        for (int x = step; x < img->width - step; x += step) {
            // 创建一个描述符
            Descriptor desc = create_descriptor(SIFT_DESC_SIZE);
            desc.x = (float)x;
            desc.y = (float)y;

            // SIFT描述符参数
            int grid_size = 4;  // 4x4网格
            int cell_size = 4;  // 每个单元格4x4像素
            int bins = 8;       // 8个方向

            // 以(x,y)为中心提取描述符
            int half_width = grid_size * cell_size / 2;

            for (int grid_y = 0; grid_y < grid_size; grid_y++) {
                for (int grid_x = 0; grid_x < grid_size; grid_x++) {
                    // 计算当前单元格的起始位置
                    int cell_start_x = x - half_width + grid_x * cell_size;
                    int cell_start_y = y - half_width + grid_y * cell_size;

                    // 遍历单元格内的每个像素
                    for (int cell_y = 0; cell_y < cell_size; cell_y++) {
                        for (int cell_x = 0; cell_x < cell_size; cell_x++) {
                            int px = cell_start_x + cell_x;
                            int py = cell_start_y + cell_y;

                            // 确保在图像边界内
                            if (px >= 0 && px < img->width && py >= 0 && py < img->height) {
                                float mag = get_pixel_gray(&magnitude, px, py);
                                float angle = get_pixel_gray(&orientation, px, py);

                                // 将角度归一化到0-2PI
                                if (angle < 0) angle += 2.0f * M_PI;

                                // 计算角度所在的bin
                                int bin = (int)(bins * angle / (2.0f * M_PI)) % bins;

                                // 计算对应的描述符索引
                                int descriptor_idx = (grid_y * grid_size + grid_x) * bins + bin;

                                // 累加梯度幅值
                                desc.data[descriptor_idx] += mag;
                            }
                        }
                    }
                }
            }

            // 归一化描述符
            float sum = 0.0f;
            for (int i = 0; i < SIFT_DESC_SIZE; i++) {
                sum += desc.data[i] * desc.data[i];
            }

            if (sum > 0) {
                float norm = 1.0f / sqrtf(sum);
                for (int i = 0; i < SIFT_DESC_SIZE; i++) {
                    desc.data[i] *= norm;
                }
            }

            add_descriptor(&list, desc);
        }
    }

    free_gray_image(&gray);
    free_gray_image(&magnitude);
    free_gray_image(&orientation);

    return list;
}