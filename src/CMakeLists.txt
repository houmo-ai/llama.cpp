llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-hybrid-iswa.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampler.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            houmo/houmo-llmodel.cpp
            houmo/houmo-gptoss.cpp
            houmo/houmo-qwenllm.cpp
            houmo/houmo-qwen35llm.cpp
            houmo/houmo-vllmodel.cpp
            houmo/houmo-embed.cpp
            houmo/houmo-embedding_layer.cpp
            ${LLAMA_MODELS_SOURCES}
            )

# set_target_properties(llama PROPERTIES
#     VERSION ${LLAMA_INSTALL_VERSION}
#     SOVERSION 0
# )

target_include_directories(llama PRIVATE . houmo)
target_include_directories(llama PUBLIC ../include ../vendor ../common ${HOUMO_TCIM_RUNTIME_PATH}/include)
target_compile_features   (llama PUBLIC cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml ${HOUMO_RUNTIME_LIBRARIES})

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
