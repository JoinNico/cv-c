#include "kmeans.h"

// 随机初始化聚类中心
static void initialize_centers(float** data, int num_points, int dim, int num_clusters, float** centers) {
    // 使用Forgy方法：随机选择数据点作为初始中心
    init_random();

    // 跟踪已选择的点索引，确保不会选择同一个点两次
    int* selected = (int*)malloc(num_points * sizeof(int));
    memset(selected, 0, num_points * sizeof(int));

    for (int i = 0; i < num_clusters; i++) {
        int idx;
        // 寻找一个尚未被选择的点
        do {
            idx = random_int(0, num_points - 1);
        } while (selected[idx]);

        // 标记为已选择
        selected[idx] = 1;

        // 复制数据点到中心
        for (int j = 0; j < dim; j++) {
            centers[i][j] = data[idx][j];
        }
    }

    free(selected);
}

// 为每个数据点分配最近的簇
static void assign_clusters(float** data, int num_points, int dim, int num_clusters, float** centers, int* assignments) {
    for (int i = 0; i < num_points; i++) {
        float min_dist = FLT_MAX;
        int best_cluster = 0;

        // 找到最近的聚类中心
        for (int j = 0; j < num_clusters; j++) {
            float dist = euclidean_distance(data[i], centers[j], dim);
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = j;
            }
        }

        assignments[i] = best_cluster;
    }
}

// 更新聚类中心
static int update_centers(float** data, int num_points, int dim, int num_clusters, int* assignments, float** centers) {
    int changed = 0;

    // 为每个聚类创建临时存储
    float** new_centers = allocate_float_matrix(num_clusters, dim);
    int* counts = (int*)malloc(num_clusters * sizeof(int));
    memset(counts, 0, num_clusters * sizeof(int));

    // 累加每个簇的所有点
    for (int i = 0; i < num_points; i++) {
        int cluster = assignments[i];
        counts[cluster]++;

        for (int j = 0; j < dim; j++) {
            new_centers[cluster][j] += data[i][j];
        }
    }

    // 计算每个簇的平均值
    for (int i = 0; i < num_clusters; i++) {
        if (counts[i] > 0) {
            for (int j = 0; j < dim; j++) {
                float new_val = new_centers[i][j] / (float)counts[i];

                // 检查中心是否移动
                if (fabsf(new_val - centers[i][j]) > 1e-4) {
                    changed = 1;
                }

                centers[i][j] = new_val;
            }
        }
    }

    free_float_matrix(new_centers, num_clusters);
    free(counts);

    return changed;
}

// 执行K-means聚类
KMeansResult kmeans_cluster(float** data, int num_points, int dim, int num_clusters, int max_iter) {
    KMeansResult result;
    result.num_clusters = num_clusters;
    result.dim = dim;
    result.num_points = num_points;

    // 分配内存
    result.centers = allocate_float_matrix(num_clusters, dim);
    result.assignments = (int*)malloc(num_points * sizeof(int));

    // 随机初始化聚类中心
    initialize_centers(data, num_points, dim, num_clusters, result.centers);

    // 迭代更新
    int iteration = 0;
    int changed = 1;

    while (changed && iteration < max_iter) {
        // 为每个数据点分配簇
        assign_clusters(data, num_points, dim, num_clusters, result.centers, result.assignments);

        // 更新簇中心
        changed = update_centers(data, num_points, dim, num_clusters, result.assignments, result.centers);

        iteration++;
    }

    printf("K-means converged after %d iterations\n", iteration);

    return result;
}

// 释放K-means结果
void free_kmeans_result(KMeansResult* result) {
    if (result) {
        free_float_matrix(result->centers, result->num_clusters);
        free(result->assignments);
        result->centers = NULL;
        result->assignments = NULL;
    }
}

// 从描述符列表构建码本
Codebook build_codebook(DescriptorList* descriptors, int num_clusters) {
    Codebook codebook;
    codebook.num_clusters = num_clusters;
    codebook.dim = descriptors->count > 0 ? descriptors->descriptors[0].length : 0;

    if (descriptors->count == 0 || codebook.dim == 0) {
        fprintf(stderr, "Error: Cannot build codebook from empty descriptor list\n");
        codebook.centers = NULL;
        return codebook;
    }

    // 将所有描述符收集到一个矩阵中
    float** data = allocate_float_matrix(descriptors->count, codebook.dim);
    for (int i = 0; i < descriptors->count; i++) {
        for (int j = 0; j < codebook.dim; j++) {
            data[i][j] = descriptors->descriptors[i].data[j];
        }
    }

    // 执行K-means聚类
    printf("Building codebook with %d clusters from %d descriptors\n", num_clusters, descriptors->count);
    KMeansResult kmeans = kmeans_cluster(data, descriptors->count, codebook.dim, num_clusters, 100);

    // 复制聚类中心作为码本
    codebook.centers = allocate_float_matrix(num_clusters, codebook.dim);
    for (int i = 0; i < num_clusters; i++) {
        for (int j = 0; j < codebook.dim; j++) {
            codebook.centers[i][j] = kmeans.centers[i][j];
        }
    }

    // 清理
    free_kmeans_result(&kmeans);
    free_float_matrix(data, descriptors->count);

    return codebook;
}

// 释放码本
void free_codebook(Codebook* codebook) {
    if (codebook && codebook->centers) {
        free_float_matrix(codebook->centers, codebook->num_clusters);
        codebook->centers = NULL;
        codebook->num_clusters = 0;
        codebook->dim = 0;
    }
}

// 查找最近的中心
int find_nearest_center(float* desc, Codebook* codebook) {
    float min_dist = FLT_MAX;
    int best_center = 0;

    for (int i = 0; i < codebook->num_clusters; i++) {
        float dist = euclidean_distance(desc, codebook->centers[i], codebook->dim);
        if (dist < min_dist) {
            min_dist = dist;
            best_center = i;
        }
    }

    return best_center;
}

// 将描述符量化为直方图
float* quantize_descriptors(DescriptorList* descriptors, Codebook* codebook) {
    float* histogram = allocate_float_array(codebook->num_clusters);

    // 遍历所有描述符
    for (int i = 0; i < descriptors->count; i++) {
        // 找到最近的码本中心
        int center = find_nearest_center(descriptors->descriptors[i].data, codebook);

        // 增加直方图中对应的bin计数
        histogram[center]++;
    }

    // 归一化直方图
    normalize_vector(histogram, codebook->num_clusters);

    return histogram;
}