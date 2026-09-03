cmake_minimum_required(VERSION 3.16)
project(yolov8_seg_camera LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui dnn)
find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED IMPORTED_TARGET
    gstreamer-1.0 gstreamer-video-1.0 gstreamer-app-1.0)

set(SPACEMIT_ORT_ROOT "" CACHE PATH
    "SpaceMIT ONNX Runtime prefix containing include/ and lib/")
if(SPACEMIT_ORT_ROOT)
    find_path(SPACEMIT_ORT_INCLUDE_DIR NAMES onnxruntime_cxx_api.h
        PATHS "${SPACEMIT_ORT_ROOT}/include" NO_DEFAULT_PATH)
    find_path(SPACEMIT_ORT_ENV_INCLUDE_DIR NAMES spacemit_ort_env.h
        PATHS "${SPACEMIT_ORT_ROOT}/include" NO_DEFAULT_PATH)
    find_library(SPACEMIT_ORT_LIBRARY NAMES onnxruntime
        PATHS "${SPACEMIT_ORT_ROOT}/lib"
              "${SPACEMIT_ORT_ROOT}/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
        NO_DEFAULT_PATH)
    find_library(SPACEMIT_EP_LIBRARY NAMES spacemit_ep
        PATHS "${SPACEMIT_ORT_ROOT}/lib"
              "${SPACEMIT_ORT_ROOT}/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
        NO_DEFAULT_PATH)
else()
    find_path(SPACEMIT_ORT_INCLUDE_DIR NAMES onnxruntime_cxx_api.h)
    find_path(SPACEMIT_ORT_ENV_INCLUDE_DIR NAMES spacemit_ort_env.h)
    find_library(SPACEMIT_ORT_LIBRARY NAMES onnxruntime)
    find_library(SPACEMIT_EP_LIBRARY NAMES spacemit_ep)
endif()

if(NOT SPACEMIT_ORT_INCLUDE_DIR OR NOT SPACEMIT_ORT_ENV_INCLUDE_DIR OR
   NOT SPACEMIT_ORT_LIBRARY OR NOT SPACEMIT_EP_LIBRARY)
    message(FATAL_ERROR
        "SpaceMIT ONNX Runtime development files were not found.\n"
        "Configure -DSPACEMIT_ORT_ROOT=<prefix> when they are outside system paths.")
endif()

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/config.cpp
    src/affinity.cpp
    src/gstreamer_camera.cpp
    src/opencl_preprocess.cpp
    src/yolov8_seg_detector.cpp
    src/overlay.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${SPACEMIT_ORT_INCLUDE_DIR}
    ${SPACEMIT_ORT_ENV_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    OpenCL::OpenCL
    ${SPACEMIT_ORT_LIBRARY}
    ${SPACEMIT_EP_LIBRARY}
    Threads::Threads
    PkgConfig::GSTREAMER
    dl)

target_compile_options(${PROJECT_NAME} PRIVATE -O3 -ffast-math -Wall -Wextra -Wpedantic)

include(CTest)
if(BUILD_TESTING)
    add_executable(affinity_test tests/affinity_test.cpp src/affinity.cpp)
    target_include_directories(affinity_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME affinity_test COMMAND affinity_test)

    add_executable(postprocess_test tests/postprocess_test.cpp)
    target_include_directories(postprocess_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(postprocess_test PRIVATE ${OpenCV_LIBS})
    add_test(NAME postprocess_test COMMAND postprocess_test)

    add_executable(latest_queue_test tests/latest_queue_test.cpp)
    target_include_directories(latest_queue_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME latest_queue_test COMMAND latest_queue_test)
endif()
