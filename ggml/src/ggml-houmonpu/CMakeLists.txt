cmake_minimum_required(VERSION 3.14)
project(ggml-houmonpu CXX)

set(GGML_HOUMONPU_SOURCES ggml-houmonpu.cpp)

# 静态库模式
add_library(houmonpu STATIC ${GGML_HOUMONPU_SOURCES})

target_include_directories(houmonpu
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(houmonpu
    PRIVATE
        ggml-base
        ${CMAKE_DL_LIBS}
)

target_compile_definitions(houmonpu
    PRIVATE
        GGML_BUILD
        GGML_USE_HOUMONPU
)

set_target_properties(houmonpu
    PROPERTIES
        OUTPUT_NAME "ggml-houmonpu"
        POSITION_INDEPENDENT_CODE ON
)

if (${BUILD_SHARED_LIBS})
    target_compile_definitions(houmonpu PUBLIC GGML_BACKEND_BUILD)
    target_compile_definitions(houmonpu PUBLIC  GGML_BACKEND_SHARED)
endif()
