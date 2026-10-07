cmake_minimum_required(VERSION 3.28)

# The current implementation is compiled only for sm_120a. Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  message(FATAL_ERROR
    "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI, server and perplexity evaluator" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)
option(NINFER_BUILD_XIAOMI_QUALIFICATION "Build Xiaomi OCR kernel oracle qualification" OFF)
option(NINFER_BUILD_ASR_QUALIFICATION "Build Qwen3-ASR independent operator qualification" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS ninfer_link=1)
  set(CMAKE_JOB_POOL_LINK ninfer_link)
endif()

# Tests also exercise product input, logging and protocol adapters.
set(NINFER_BUILD_PRODUCT_SUPPORT OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_PRODUCT_SUPPORT ON)
endif()

include(cmake/Dependencies.cmake)
include(cmake/NinferTargets.cmake)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
if(NINFER_BUILD_XIAOMI_QUALIFICATION)
  add_executable(xiaomi-ocr-qualify tools/xiaomi_ocr/qualify_attention.cu)
  ninfer_internal_includes(xiaomi-ocr-qualify)
  target_link_libraries(xiaomi-ocr-qualify PRIVATE ninfer_ops CUDA::cudart)
  add_executable(xiaomi-ocr-qualify-small-ops tools/xiaomi_ocr/qualify_small_ops.cu)
  ninfer_internal_includes(xiaomi-ocr-qualify-small-ops)
  target_link_libraries(xiaomi-ocr-qualify-small-ops PRIVATE ninfer_ops CUDA::cudart)
  add_executable(xiaomi-ocr-qualify-causal tools/xiaomi_ocr/qualify_causal_decode.cu)
  ninfer_internal_includes(xiaomi-ocr-qualify-causal)
  target_link_libraries(xiaomi-ocr-qualify-causal PRIVATE ninfer_ops CUDA::cudart)
endif()
if(NINFER_BUILD_ASR_QUALIFICATION)
  add_executable(asr-qualify tools/qwen3_asr/qualify.cu)
  ninfer_internal_includes(asr-qualify)
  target_link_libraries(asr-qualify PRIVATE ninfer_ops CUDA::cudart)
endif()
