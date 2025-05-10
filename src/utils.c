#include "utils.h"

// 内存分配函数
float* allocate_float_array(int size) {
    float* array = (float*)malloc(size * sizeof(float));
    if (!array) {
        fprintf(stderr, "Error: Memory allocation failed for float array\n");
        exit(EXIT_FAILURE);
    }
    memset(array, 0, size * sizeof(float));
    return array;
}

float** allocate_float_matrix(int rows, int cols) {
    float** matrix = (float**)malloc(rows * sizeof(float*));
    if (!matrix) {
        fprintf(stderr, "Error: Memory allocation failed for matrix rows\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++) {
        matrix[i] = allocate_float_array(cols);
    }
    return matrix;
}

void free_float_array(float* array) {
    if (array) {
        free(array);
    }
}

void free_float_matrix(float** matrix, int rows) {
    if (matrix) {
        for (int i = 0; i < rows; i++) {
            free_float_array(matrix[i]);
        }
        free(matrix);
    }
}

// 描述符操作函数
Descriptor create_descriptor(int length) {
    Descriptor desc;
    desc.length = length;
    desc.data = allocate_float_array(length);
    desc.x = 0.0f;
    desc.y = 0.0f;
    return desc;
}

void free_descriptor(Descriptor* desc) {
    if (desc && desc->data) {
        free_float_array(desc->data);
        desc->data = NULL;
        desc->length = 0;
    }
}

DescriptorList create_descriptor_list(int initial_capacity) {
    DescriptorList list;
    list.count = 0;
    list.descriptors = (Descriptor*)malloc(initial_capacity * sizeof(Descriptor));
    if (!list.descriptors) {
        fprintf(stderr, "Error: Memory allocation failed for descriptor list\n");
        exit(EXIT_FAILURE);
    }
    return list;
}

void free_descriptor_list(DescriptorList* list) {
    if (list && list->descriptors) {
        for (int i = 0; i < list->count; i++) {
            free_descriptor(&list->descriptors[i]);
        }
        free(list->descriptors);
        list->descriptors = NULL;
        list->count = 0;
    }
}

void add_descriptor(DescriptorList* list, Descriptor desc) {
    // 动态扩容，这里简化处理，实际应用中应该更智能地扩容
    Descriptor* new_array = (Descriptor*)realloc(list->descriptors,
                                                 (list->count + 1) * sizeof(Descriptor));
    if (!new_array) {
        fprintf(stderr, "Error: Memory reallocation failed for descriptor list\n");
        exit(EXIT_FAILURE);
    }

    list->descriptors = new_array;
    list->descriptors[list->count] = desc;
    list->count++;
}

// 数学工具函数
float euclidean_distance(float* v1, float* v2, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

float chi_square_distance(float* v1, float* v2, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        if (v1[i] + v2[i] > 0) {
            float diff = v1[i] - v2[i];
            sum += (diff * diff) / (v1[i] + v2[i]);
        }
    }
    return 0.5f * sum;
}

int min_int(int a, int b) {
    return (a < b) ? a : b;
}

float min_float(float a, float b) {
    return (a < b) ? a : b;
}

// 随机数生成
void init_random() {
    srand((unsigned int)time(NULL));
}

int random_int(int min, int max) {
    return min + rand() % (max - min + 1);
}

float random_float(float min, float max) {
    float scale = rand() / (float)RAND_MAX;
    return min + scale * (max - min);
}

// 向量操作
void normalize_vector(float* vec, int length) {
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        sum += vec[i];
    }

    if (sum > 0) {
        for (int i = 0; i < length; i++) {
            vec[i] /= sum;
        }
    }
}

void vector_add(float* result, float* v1, float* v2, int length) {
    for (int i = 0; i < length; i++) {
        result[i] = v1[i] + v2[i];
    }
}

void vector_multiply_scalar(float* result, float* vec, float scalar, int length) {
    for (int i = 0; i < length; i++) {
        result[i] = vec[i] * scalar;
    }
}

void print_vector(float* vec, int length) {
    printf("[");
    for (int i = 0; i < length; i++) {
        printf("%.4f", vec[i]);
        if (i < length - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

// 文件操作
unsigned char* read_file(const char* filename, size_t* size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    fseek(file, 0, SEEK_SET);

    unsigned char* buffer = (unsigned char*)malloc(*size);
    if (!buffer) {
        fprintf(stderr, "Error: Memory allocation failed for file buffer\n");
        fclose(file);
        return NULL;
    }

    size_t bytes_read = fread(buffer, 1, *size, file);
    fclose(file);

    if (bytes_read != *size) {
        fprintf(stderr, "Error: Could not read entire file %s\n", filename);
        free(buffer);
        return NULL;
    }

    return buffer;
}

int write_file(const char* filename, unsigned char* data, size_t size) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
        return 0;
    }

    size_t bytes_written = fwrite(data, 1, size, file);
    fclose(file);

    if (bytes_written != size) {
        fprintf(stderr, "Error: Could not write entire data to file %s\n", filename);
        return 0;
    }

    return 1;
}