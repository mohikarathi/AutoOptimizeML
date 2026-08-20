cmake_minimum_required(VERSION 3.18)
project(autoopt_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP CXX found: ${OpenMP_CXX_FLAGS}")
endif()

# Check CUDA support
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(ENABLE_CUDA ON)
    add_compile_definitions(ENABLE_CUDA)
    message(STATUS "CUDA Compiler enabled: ${CMAKE_CUDA_COMPILER}")
else()
    set(ENABLE_CUDA OFF)
    message(STATUS "CUDA Compiler not active, compiling OpenMP C++ CPU native backend.")
endif()

# Find Python and PyBind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Get extension suffix from python
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get PyBind11 includes from python
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "-I" "" PYBIND11_INCLUDE_DIRS "${PYBIND11_INCLUDES}")
separate_arguments(PYBIND11_INCLUDE_DIRS)

set(SOURCES
    src/tensor_ops_cpu.cpp
    src/bindings.cpp
)

if(ENABLE_CUDA)
    list(APPEND SOURCES src/tensor_ops_cuda.cu)
endif()

add_library(autoopt_native MODULE ${SOURCES})
target_include_directories(autoopt_native PRIVATE
    include
    /usr/local/cuda-12.2/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${PYBIND11_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

set_target_properties(autoopt_native PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_EXT_SUFFIX}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../autoopt"
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(autoopt_native PRIVATE OpenMP::OpenMP_CXX)
endif()

if(ENABLE_CUDA)
    target_link_directories(autoopt_native PRIVATE /usr/local/cuda-12.2/lib64)
    target_link_libraries(autoopt_native PRIVATE cudart)
endif()
