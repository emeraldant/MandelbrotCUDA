cmake_minimum_required(VERSION 3.18)
project(CudaMandelbrot LANGUAGES CXX CUDA)

# Use modern C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Set SFML paths manually
set(SFML_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/SFML-3.0.0")
set(SFML_INCLUDE_DIR "${SFML_ROOT}/include")
set(SFML_LIBRARY_DIR "${SFML_ROOT}/lib")

# Add source files
add_executable(CudaMandelbrot
    src/main.cpp
    src/Mandelbrot.cpp
    src/Renderer.cpp
    src/CudaKernels.cu
)

# Include directories
target_include_directories(CudaMandelbrot PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${SFML_INCLUDE_DIR}
)

# Link directories
target_link_directories(CudaMandelbrot PRIVATE
    ${SFML_LIBRARY_DIR}
)

# Link libraries
target_link_libraries(CudaMandelbrot PRIVATE 
    "sfml-system"
    "sfml-window"
    "sfml-graphics"
)

# Set output directories
set_target_properties(CudaMandelbrot PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
)

# Copy SFML DLLs to output directory
add_custom_command(TARGET CudaMandelbrot POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${SFML_ROOT}/bin"
    $<TARGET_FILE_DIR:CudaMandelbrot>
)
