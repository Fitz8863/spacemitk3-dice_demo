cmake_minimum_required(VERSION 3.16)
project(k3_circle_detect CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# 系统 OpenCV 4.10 已在 K3 (Bianbu) 上装好；如需 SDK 里的 opencv 可传
#   -DOpenCV_DIR=/opt/opencv-spacemit/lib/cmake/opencv4
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)

# RTSP 推流：GStreamer（appsrc / videoconvert / spacemith264enc / rtspclientsink）
# 与 dice-game 的 yolov8_segdetect 用同一套依赖。
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED IMPORTED_TARGET
    gstreamer-1.0 gstreamer-video-1.0 gstreamer-app-1.0)

add_executable(circle_detect
  src/main.cpp
  src/circle_detector.cpp
  src/config.cpp
  src/frame_source.cpp
  src/mjpeg_server.cpp
  src/rtsp_streamer.cpp
)

# MJPEG 预览服务用 POSIX socket，需要 threads
find_package(Threads REQUIRED)
target_include_directories(circle_detect PRIVATE src)
target_link_libraries(circle_detect PRIVATE
    ${OpenCV_LIBS} Threads::Threads PkgConfig::GSTREAMER)
target_compile_options(circle_detect PRIVATE -Wall -Wextra -Wno-unused-parameter)

message(STATUS "OpenCV ${OpenCV_VERSION} @ ${OpenCV_DIR}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
