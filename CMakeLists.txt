cmake_minimum_required(VERSION 3.16)
project(zipformer_streaming_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------
# onnxruntime（spacemit-onnxruntime 2.0.6 = ORT 1.24.2 + SpaceMIT EP）
# ---------------------------------------------------------------
find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h
          PATHS /usr/include /usr/local/include)
find_library(ONNXRUNTIME_LIB NAMES onnxruntime
             PATHS /usr/local/lib /usr/lib)
find_library(SPACEMIT_EP_LIB NAMES spacemit_ep
             PATHS /usr/local/lib /usr/lib)

# ---------------------------------------------------------------
# kaldi-native-fbank：直接复用 model-zoo-asr 编译产物（静态库）
# ---------------------------------------------------------------
set(_ASR_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../model-zoo-asr/build")
find_library(KNF_LIB NAMES kaldi-native-fbank-core
             PATHS ${_ASR_BUILD_DIR}/lib)
find_library(KISSFFT_LIB NAMES kissfft-float
             PATHS ${_ASR_BUILD_DIR}/lib)
find_path(KNF_INC_DIR NAMES online-feature.h
          PATHS ${_ASR_BUILD_DIR}/kaldi-native-fbank-build/include/kaldi-native-fbank/csrc)
find_path(KNF_INC_ROOT NAMES kaldi-native-fbank/csrc/feature-fbank.h
          PATHS ${_ASR_BUILD_DIR}/kaldi-native-fbank-build/include)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "onnxruntime 头文件/库没找到")
endif()
if(NOT KNF_LIB OR NOT KISSFFT_LIB OR NOT KNF_INC_DIR)
    message(FATAL_ERROR
            "kaldi-native-fbank 预编译产物没找到（期望 ${_ASR_BUILD_DIR}/lib）。"
            " 请先编译 model-zoo-asr，或手动指定 KNF_LIB/KISSFFT_LIB/KNF_INC_DIR。")
endif()

# ---------------------------------------------------------------
# sndfile（读 wav）
# ---------------------------------------------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)

# ---------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------
add_executable(stream_asr
    src/zipformer_streaming.cc
    src/stream_asr_main.cc)

target_include_directories(stream_asr PRIVATE
    src
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${KNF_INC_DIR}
    ${KNF_INC_ROOT}
    ${SNDFILE_INCLUDE_DIRS})

target_link_libraries(stream_asr PRIVATE
    ${ONNXRUNTIME_LIB}
    ${KNF_LIB}
    ${KISSFFT_LIB}
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
