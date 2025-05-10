cmake_minimum_required(VERSION 3.23)
project(cv_c C)

set(CMAKE_C_STANDARD 11)

include_directories(inc)

add_executable(cv_c
        inc/kmeans.h
        inc/sift.h
        inc/spm.h
        inc/utils.h
        inc/image.h
        src/image.c
        src/kmeans.c
        src/sift.c
        src/spm.c
        src/utils.c)
