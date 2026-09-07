# FetchFunAsrModel.cmake - Fetch Fun-ASR model to cache
# Sets: FUNASR_MODEL_DIR
# Note: This model is used by llama-server, not directly by the ASR library

if(DEFINED _FETCH_FUNASR_MODEL_LOADED)
  return()
endif()
set(_FETCH_FUNASR_MODEL_LOADED ON)

set(_FUNASR_MODEL_URL "https://archive.spacemit.com/spacemit-ai/model_zoo/asr/fun-asr-nano-2512-qq-q4km.tar.gz")
set(_FUNASR_MODEL_NAME "fun-asr-nano-2512-qq-q4km")
set(_FUNASR_ARCHIVE_SUBDIR "fun-asr-nano-2512-qq-q4km")
set(_FUNASR_REQUIRED_FILES
  "qwen3-0.6b-q4km.gguf"
  "frontend.q.onnx"
  "backend.q.onnx"
  "config.json"
)

if(DEFINED ENV{HOME})
  set(_FUNASR_CACHE_ROOT "$ENV{HOME}/.cache/models/asr")
else()
  set(_FUNASR_CACHE_ROOT "${CMAKE_BINARY_DIR}/.cache/models/asr")
endif()

set(_FUNASR_MODEL_DIR "${_FUNASR_CACHE_ROOT}/${_FUNASR_MODEL_NAME}")

set(_need_download OFF)
foreach(_file IN LISTS _FUNASR_REQUIRED_FILES)
  if(NOT EXISTS "${_FUNASR_MODEL_DIR}/${_file}")
    set(_need_download ON)
    break()
  endif()
endforeach()

if(_need_download)
  if(DEFINED ASR_MODEL_FETCH_OFF AND ASR_MODEL_FETCH_OFF)
    message(WARNING "Fun-ASR model not found at ${_FUNASR_MODEL_DIR}, fetch disabled (ASR_MODEL_FETCH_OFF).")
    return()
  endif()

  message(STATUS "Fetching Fun-ASR model to ${_FUNASR_MODEL_DIR} ...")
  file(MAKE_DIRECTORY "${_FUNASR_MODEL_DIR}")
  set(_archive_path "${_FUNASR_MODEL_DIR}/${_FUNASR_MODEL_NAME}.tar.gz")

  file(DOWNLOAD
    "${_FUNASR_MODEL_URL}"
    "${_archive_path}"
    SHOW_PROGRESS
    STATUS _download_status
    TLS_VERIFY OFF
  )

  list(GET _download_status 0 _download_code)
  if(NOT _download_code EQUAL 0)
    list(GET _download_status 1 _download_error)
    message(FATAL_ERROR "Failed to download Fun-ASR model: ${_download_error}")
  endif()

  message(STATUS "Extracting Fun-ASR model...")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf "${_archive_path}"
    WORKING_DIRECTORY "${_FUNASR_MODEL_DIR}"
    RESULT_VARIABLE _extract_result
  )
  if(NOT _extract_result EQUAL 0)
    message(FATAL_ERROR "Failed to extract Fun-ASR model")
  endif()

  if(EXISTS "${_FUNASR_MODEL_DIR}/${_FUNASR_ARCHIVE_SUBDIR}")
    file(GLOB _subdir_files "${_FUNASR_MODEL_DIR}/${_FUNASR_ARCHIVE_SUBDIR}/*")
    foreach(_file IN LISTS _subdir_files)
      get_filename_component(_filename "${_file}" NAME)
      file(RENAME "${_file}" "${_FUNASR_MODEL_DIR}/${_filename}")
    endforeach()
    file(REMOVE_RECURSE "${_FUNASR_MODEL_DIR}/${_FUNASR_ARCHIVE_SUBDIR}")
  endif()

  file(REMOVE "${_archive_path}")
  message(STATUS "Fun-ASR model ready at ${_FUNASR_MODEL_DIR}")
else()
  message(STATUS "Fun-ASR model found at ${_FUNASR_MODEL_DIR}")
endif()

set(FUNASR_MODEL_DIR "${_FUNASR_MODEL_DIR}" CACHE PATH "Fun-ASR model directory" FORCE)
