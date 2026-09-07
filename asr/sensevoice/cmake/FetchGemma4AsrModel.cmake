# FetchGemma4AsrModel.cmake - Fetch Gemma4 ASR model to cache
# Sets: GEMMA4_ASR_MODEL_DIR
# Note: This model is used by llama-server, not directly by the ASR library

if(DEFINED _FETCH_GEMMA4_ASR_MODEL_LOADED)
  return()
endif()
set(_FETCH_GEMMA4_ASR_MODEL_LOADED ON)

set(_GEMMA4_ASR_MODEL_URL "https://archive.spacemit.com/spacemit-ai/model_zoo/asr/gemma4-asr-E2B-q40.tar.gz")
set(_GEMMA4_ASR_MODEL_NAME "gemma4-asr-E2B-q40")
set(_GEMMA4_ASR_ARCHIVE_SUBDIR "gemma4-asr-E2B-q40")
set(_GEMMA4_ASR_REQUIRED_FILES
  "audio_encoder.q.onnx"
  "config.json"
  "gemma-4-E2B-it-Q4_0-plproj-Q4_0-combined.gguf"
)

if(DEFINED ENV{HOME})
  set(_GEMMA4_ASR_CACHE_ROOT "$ENV{HOME}/.cache/models/asr")
else()
  set(_GEMMA4_ASR_CACHE_ROOT "${CMAKE_BINARY_DIR}/.cache/models/asr")
endif()

set(_GEMMA4_ASR_MODEL_DIR "${_GEMMA4_ASR_CACHE_ROOT}/${_GEMMA4_ASR_MODEL_NAME}")

set(_need_download OFF)
foreach(_file IN LISTS _GEMMA4_ASR_REQUIRED_FILES)
  if(NOT EXISTS "${_GEMMA4_ASR_MODEL_DIR}/${_file}")
    set(_need_download ON)
    break()
  endif()
endforeach()

if(_need_download)
  if(DEFINED ASR_MODEL_FETCH_OFF AND ASR_MODEL_FETCH_OFF)
    message(WARNING "Gemma4 ASR model not found at ${_GEMMA4_ASR_MODEL_DIR}, fetch disabled (ASR_MODEL_FETCH_OFF).")
    return()
  endif()

  message(STATUS "Fetching Gemma4 ASR model to ${_GEMMA4_ASR_MODEL_DIR} ...")
  file(MAKE_DIRECTORY "${_GEMMA4_ASR_MODEL_DIR}")
  set(_archive_path "${_GEMMA4_ASR_MODEL_DIR}/${_GEMMA4_ASR_MODEL_NAME}.tar.gz")

  file(DOWNLOAD
    "${_GEMMA4_ASR_MODEL_URL}"
    "${_archive_path}"
    SHOW_PROGRESS
    STATUS _download_status
    TLS_VERIFY OFF
  )

  list(GET _download_status 0 _download_code)
  if(NOT _download_code EQUAL 0)
    list(GET _download_status 1 _download_error)
    message(FATAL_ERROR "Failed to download Gemma4 ASR model: ${_download_error}")
  endif()

  message(STATUS "Extracting Gemma4 ASR model...")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf "${_archive_path}"
    WORKING_DIRECTORY "${_GEMMA4_ASR_MODEL_DIR}"
    RESULT_VARIABLE _extract_result
  )
  if(NOT _extract_result EQUAL 0)
    message(FATAL_ERROR "Failed to extract Gemma4 ASR model")
  endif()

  if(EXISTS "${_GEMMA4_ASR_MODEL_DIR}/${_GEMMA4_ASR_ARCHIVE_SUBDIR}")
    file(GLOB _subdir_files "${_GEMMA4_ASR_MODEL_DIR}/${_GEMMA4_ASR_ARCHIVE_SUBDIR}/*")
    foreach(_file IN LISTS _subdir_files)
      get_filename_component(_filename "${_file}" NAME)
      file(RENAME "${_file}" "${_GEMMA4_ASR_MODEL_DIR}/${_filename}")
    endforeach()
    file(REMOVE_RECURSE "${_GEMMA4_ASR_MODEL_DIR}/${_GEMMA4_ASR_ARCHIVE_SUBDIR}")
  endif()

  file(REMOVE "${_archive_path}")
  message(STATUS "Gemma4 ASR model ready at ${_GEMMA4_ASR_MODEL_DIR}")
else()
  message(STATUS "Gemma4 ASR model found at ${_GEMMA4_ASR_MODEL_DIR}")
endif()

set(GEMMA4_ASR_MODEL_DIR "${_GEMMA4_ASR_MODEL_DIR}" CACHE PATH "Gemma4 ASR model directory" FORCE)
