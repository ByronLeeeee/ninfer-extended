find_package(CUDAToolkit REQUIRED)
# Native ASR uses BF16 cuDNN convolution and a cuBLAS reference route.
find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS "${CUDNN_ROOT}/include" REQUIRED)
find_library(CUDNN_LIBRARY NAMES cudnn libcudnn.so.9 HINTS "${CUDNN_ROOT}/lib" "${CUDNN_ROOT}/lib64" REQUIRED)
find_library(NINFER_CUBLAS_LIBRARY NAMES cublas libcublas.so.13 HINTS "${CUBLAS_ROOT}/lib" "${CUBLAS_ROOT}/lib64" "${CUDAToolkit_LIBRARY_DIR}" REQUIRED)
add_library(ninfer_cudnn INTERFACE)
target_include_directories(ninfer_cudnn INTERFACE "${CUDNN_INCLUDE_DIR}")
target_link_libraries(ninfer_cudnn INTERFACE "${CUDNN_LIBRARY}" "${NINFER_CUBLAS_LIBRARY}")
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED IMPORTED_TARGET libpcre2-8)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
  libavformat libavcodec libavutil libswscale)

# Repository-pinned header dependencies. No configure-time downloads.
add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

# Source base for the custom-template frontend; consumers will link it explicitly.
add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)

if(NINFER_BUILD_PRODUCT_SUPPORT)
  # Media acquisition uses CURLOPT_PROTOCOLS_STR and CURLOPT_REDIR_PROTOCOLS_STR,
  # introduced in libcurl 7.85 (not merely the version of the maintainer environment).
  pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
  add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  add_subdirectory(third_party/spdlog)
endif()
