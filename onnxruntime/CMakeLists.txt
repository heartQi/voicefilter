cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project("voicefilter-onnx")
set(target "voicefilter-onnx")

set (CMAKE_CXX_STANDARD 20)

add_definitions(-DROOT_DIR="${CMAKE_CURRENT_LIST_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

list(APPEND VOICE_FILTER_SRC_FILES
  voice_filter.h
  voice_filter_i.h
  voice_filter.cpp
  librosa/librosa.h
)

if (APPLE)
  set(ONNXRUNTIME_ROOT ${CMAKE_CURRENT_LIST_DIR}/macos)
else()
  set(ONNXRUNTIME_ROOT ${CMAKE_CURRENT_LIST_DIR}/windows)
endif()

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}
    ${ONNXRUNTIME_ROOT}/include
    ${CMAKE_CURRENT_LIST_DIR}/librosa
)

link_directories(
    ${ONNXRUNTIME_ROOT}/lib
    ${CMAKE_CURRENT_BINARY_DIR}
)

add_library(voice_filter_shared SHARED ${VOICE_FILTER_SRC_FILES})
target_compile_definitions(voice_filter_shared PRIVATE -DVOICE_FILTER_EXPORT)
target_link_libraries(voice_filter_shared PRIVATE onnxruntime)

add_executable(${target} main.cpp wavreader.c)
target_link_libraries(${target} voice_filter_shared)
