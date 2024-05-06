cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project("voicefilter-ncnn")
set(target "voicefilter-ncnn")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/build)
set (CMAKE_CXX_STANDARD 17)

find_package(ncnn REQUIRED)
message(STATUS "ncnn found: ${ncnn_FRAMEWORK_DIRS}")
add_definitions(-DROOT_DIR="${CMAKE_CURRENT_LIST_DIR}")

list(APPEND VOICE_FILTER_SRC_FILES
  main.cpp
  voice_filter.h
  voice_filter.cpp
)

include_directories(
    ${CMAKE_CURRENT_LIST_DIR}
    $ENV{VULKAN_SDK}/include
    ${CMAKE_BINARY_DIR}
)

if (APPLE)
    file(GLOB DNN_MODEL_FILES ${CMAKE_CURRENT_LIST_DIR}/../model/*.bin ${CMAKE_CURRENT_LIST_DIR}/../model/*.param)
    list(APPEND NCNN_LIBS "-F ${ncnn_FRAMEWORK_DIRS}")
    foreach(framework IN LISTS ncnn_FRAMEWORKS)
        list(APPEND NCNN_FRAMEWORKS "${ncnn_FRAMEWORK_DIRS}/${framework}.framework")
        list(APPEND NCNN_LIBS "-framework ${framework}")
    endforeach(framework IN LISTS ncnn_FRAMEWORKS)
    list(APPEND NCNN_LIBS "-F $ENV{VULKAN_SDK}/Frameworks/" "-framework vulkan")

    list(APPEND NCNN_FRAMEWORKS
        "$ENV{VULKAN_SDK}/Frameworks/vulkan.framework"
        "$ENV{VULKAN_SDK}/lib/libMoltenVK.dylib"
        )
    list(APPEND VULKAN_ICD_FILENAMES "${CMAKE_CURRENT_LIST_DIR}/res/MoltenVK_icd.json")
    set_source_files_properties(${NCNN_FRAMEWORKS} PROPERTIES MACOSX_PACKAGE_LOCATION Frameworks)
    set_source_files_properties(${VULKAN_ICD_FILENAMES} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources/vulkan/icd.d")
    set_source_files_properties(${DNN_MODEL_FILES} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources/models/")

    include_directories("${ncnn_FRAMEWORK_DIRS}/ncnn.framework/Versions/Current/Headers/ncnn")
    list(APPEND VOICE_FILTER_SRC_FILES
      sys_utils.h 
      sys_utils.mm
    )
endif()

add_executable(${target} ${VOICE_FILTER_SRC_FILES} ${DNN_MODEL_FILES} ${NCNN_FRAMEWORKS} ${VULKAN_ICD_FILENAMES})

if (APPLE)
    set_target_properties(${target} PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/Info.plist
        XCODE_ATTRIBUTE_OTHER_CODE_SIGN_FLAGS "--deep"
        XCODE_ATTRIBUTE_LD_RUNPATH_SEARCH_PATHS "@executable_path/../Frameworks"
        MACOSX_PACKAGE_LOCATION Frameworks
        XCODE_FILE_ATTRIBUTES "CodeSignOnCopy"
    )
endif()

target_link_libraries(${target} ${NCNN_LIBS})
