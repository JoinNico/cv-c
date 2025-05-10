# CV-C 项目目录结构

```text
cv-c/
├── CMakeLists.txt              # CMake配置
├── src/
│   ├── main.c                  # 主程序入口
│   ├── image.c                 # 图像处理实现
│   ├── sift.c                  # SIFT特征实现
│   ├── kmeans.c                # K-means聚类实现
│   ├── spm.c                   # SPM算法核心
│   ├── svm.c                   # SVM分类器
│   └── utils.c                 # 工具函数
├── inc/                        # 公共头文件
│   ├── image.h
│   ├── sift.h
│   ├── kmeans.h
│   ├── spm.h
│   ├── svm.h
│   └── utils.h
├── data/                       # 数据集
│   └── cifar10/                # CIFAR-10数据
│       ├── train/              # 训练集
│       └── test/               # 测试集
└── docs/                       # 文档
    ├── API.md                  # 接口说明
    └── Design.md               # 设计文档
