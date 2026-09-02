cmake_minimum_required(VERSION 3.16)
project(zipformer_streaming_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------
# onnxruntime（spacemit-onnxruntime 2.0.6 = ORT 1.24.2 + SpaceMIT EP，系统包）
# ---------------------------------------------------------------
find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h
          PATHS /usr/include /usr/local/include)
find_library(ONNXRUNTIME_LIB NAMES onnxruntime
             PATHS /usr/local/lib /usr/lib)
find_library(SPACEMIT_EP_LIB NAMES spacemit_ep
             PATHS /usr/local/lib /usr/lib)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "onnxruntime 头文件/库没找到（需要 spacemit-onnxruntime 系统包）")
endif()

# ---------------------------------------------------------------
# sndfile（读 wav）
# ---------------------------------------------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)

# ---------------------------------------------------------------
# 第三方依赖（源码随仓库分发于 third_party/，无网络、无外部目录依赖）
#   third_party/kissfft              —— FFT，提供 kissfft 目标（knf 的依赖）
#   third_party/kaldi-native-fbank   —— fbank 特征提取，提供 kaldi-native-fbank-core
# ---------------------------------------------------------------
set(KISSFFT_DATATYPE float CACHE STRING "" FORCE)  # knf 用 float FFT
set(KISSFFT_STATIC ON CACHE BOOL "" FORCE)
set(KISSFFT_TEST OFF CACHE BOOL "" FORCE)
set(KISSFFT_TOOLS OFF CACHE BOOL "" FORCE)
set(KISSFFT_PKGCONFIG OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/kissfft)

set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
# 静态构建 knf（保存/恢复 BUILD_SHARED_LIBS，不污染外层）
set(_SAVED_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/kaldi-native-fbank)
# knf 源码内部用 "kaldi-native-fbank/csrc/xxx.h" 互相引用，
# 上游假设工程根目录下存在 kaldi-native-fbank/；这里把 third_party/ 
# 作为它的公共 include 根（PUBLIC 同时传递给 stream_asr）
target_include_directories(kaldi-native-fbank-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
set(BUILD_SHARED_LIBS ${_SAVED_BUILD_SHARED_LIBS} CACHE BOOL "" FORCE)

# ---------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------
add_executable(stream_asr
    src/zipformer_streaming.cc
    src/stream_asr_main.cc)

target_include_directories(stream_asr PRIVATE
    src
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${SNDFILE_INCLUDE_DIRS}
    # knf 内部头文件互相用 "kaldi-native-fbank/csrc/xxx.h" 引用 → third_party/
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    # 我们源码用 "online-feature.h" 直接引用 → csrc/
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/kaldi-native-fbank/csrc)

target_link_libraries(stream_asr PRIVATE
    ${ONNXRUNTIME_LIB}
    kaldi-native-fbank-core
    kissfft
    ${SNDFILE_LIBRARIES})

target_link_directories(stream_asr PRIVATE ${SNDFILE_LIBRARY_DIRS})

if(SPACEMIT_EP_LIB)
    target_compile_definitions(stream_asr PRIVATE USE_SPACEMIT_EP=1)
    target_link_libraries(stream_asr PRIVATE ${SPACEMIT_EP_LIB})
    message(STATUS "SpaceMIT EP: ${SPACEMIT_EP_LIB}")
else()
    message(WARNING "libspacemit_ep 没找到，只编译 CPU 模式")
endif()

# onnxruntime 在 /usr/local/lib，装个 rpath 免得每次设 LD_LIBRARY_PATH
set_target_properties(stream_asr PROPERTIES
    BUILD_RPATH "/usr/local/lib"
    INSTALL_RPATH "/usr/local/lib")

# ---------------------------------------------------------------
# 工具：dump_onnx（查看 onnx 输入输出）
# ---------------------------------------------------------------
add_executable(dump_onnx tools/dump_onnx.cc)
target_include_directories(dump_onnx PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(dump_onnx PRIVATE ${ONNXRUNTIME_LIB})
set_target_properties(dump_onnx PROPERTIES BUILD_RPATH "/usr/local/lib")
