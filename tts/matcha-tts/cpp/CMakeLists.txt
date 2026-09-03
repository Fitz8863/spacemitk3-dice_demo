cmake_minimum_required(VERSION 3.16)
project(matcha_tts_capi LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED SHERPA_ROOT)
  message(FATAL_ERROR "Pass -DSHERPA_ROOT=/path/to/sherpa_onnx")
endif()
set(SHERPA_INCLUDE "${SHERPA_ROOT}/include")
set(SHERPA_LIB "${SHERPA_LIB_DIR}")
if(NOT DEFINED SHERPA_LIB_DIR)
  set(SHERPA_LIB "${SHERPA_ROOT}/lib")
endif()
find_package(Threads REQUIRED)

add_executable(matcha_tts_capi matcha_tts_capi.cpp)
add_executable(matcha_tts_interactive matcha_tts_interactive.cpp)
add_executable(matcha_tts_service matcha_tts_service.cpp)
foreach(target matcha_tts_capi matcha_tts_interactive matcha_tts_service)
  target_include_directories(${target} PRIVATE "${SHERPA_INCLUDE}")
  target_link_directories(${target} PRIVATE "${SHERPA_LIB}")
  target_link_libraries(${target} PRIVATE sherpa-onnx-c-api Threads::Threads)
  set_target_properties(${target} PROPERTIES
    BUILD_RPATH "${SHERPA_LIB}"
    INSTALL_RPATH "${SHERPA_LIB}"
  )
endforeach()
