cmake_minimum_required(VERSION 3.10)
project(cv-c LANGUAGES C)

# Set C11 standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)  # Use strict standard compliance

# Configure output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Include directories
include_directories(
        ${CMAKE_SOURCE_DIR}/inc
)

# Source files
set(SRC_FILES
        src/main.c
        src/image.c
        src/sift.c
        src/kmeans.c
        src/spm.c
        src/svm.c
        src/utils.c
        )

# Build executable
add_executable(cv-c ${SRC_FILES})

# If you need to link any libraries
# target_link_libraries(cv-c PRIVATE m)  # Example for math library

# Install target (optional)
install(TARGETS cv-c DESTINATION bin)
install(DIRECTORY inc/ DESTINATION include)
install(DIRECTORY data/ DESTINATION share/cv-c/data)

# Enable testing if needed (optional)
# enable_testing()
# add_test(NAME cv-c-test COMMAND cv-c)