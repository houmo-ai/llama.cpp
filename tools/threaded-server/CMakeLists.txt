set(TARGET llama-threaded-server)

add_executable(${TARGET}
    threaded-server.cpp
    threaded-models.cpp
    threaded-models.h
    ../server/server-http.cpp
    ../server/server-models.cpp
)

install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/tools/server
    ${CMAKE_SOURCE_DIR}/tools/mtmd
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(${TARGET} PRIVATE
    server-context
    llama-ui
    llama-common
    cpp-httplib
    ${CMAKE_THREAD_LIBS_INIT}
)

target_compile_features(${TARGET} PRIVATE cxx_std_17)
