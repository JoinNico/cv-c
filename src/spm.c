#include "spm.h"
#include <math.h>
#include <stdlib.h>

// 构建空间金字塔
SPMPyramid* build_spm_pyramid(SiftFeature* features, int num_features,
                              KMeansModel* model, int level) {
    SPMPyramid* pyramid = (SPMPyramid*)malloc(sizeof(SPMPyramid));
    pyramid->level = level;
    pyramid->vocab_size = model->k;

    // 计算总直方图维度: vocab_size * sum(4^l) for l=0..level
    int total_dim = model->k;
    for (int l = 1; l <= level; l++) {
        total_dim += model->k * (1 << (2*l));
    }
    pyramid->histograms = (int*)calloc(total_dim, sizeof(int));

    // 构建每一层的直方图
    int offset = 0;
    for (int l = 0; l <= level; l++) {
        int cells = 1 << (2*l); // 4^l个cell
        int cell_hist_size = model->k * cells;

        // 计算每个cell的边界
        float cell_width = 1.0f / (1 << l);
        float cell_height = 1.0f / (1 << l);

        // 初始化当前层的直方图
        int* level_hist = pyramid->histograms + offset;

        // 对每个特征分配到对应的cell和视觉词
        for (int i = 0; i < num_features; i++) {
            SiftFeature* feat = &features[i];

            // 归一化坐标到[0,1]
            float nx = feat->x / IMAGE_WIDTH;
            float ny = feat->y / IMAGE_HEIGHT;

            // 确定cell索引
            int cell_x = (int)(nx / cell_width);
            int cell_y = (int)(ny / cell_height);
            int cell_idx = cell_y * (1 << l) + cell_x;

            // 预测视觉词
            int word = predict_kmeans_single(model, feat->descriptor);

            // 更新直方图
            level_hist[cell_idx * model->k + word]++;
        }

        offset += cell_hist_size;
    }

    return pyramid;
}