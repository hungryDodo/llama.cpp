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
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

# EdgeWeaver per-layer KV export helper
option(LLAMAEDGE_ENABLE_KV_LAYER_EXPORT "llama: enable per-layer KV export helper" OFF)
if(LLAMAEDGE_ENABLE_KV_LAYER_EXPORT)
    target_compile_definitions(llama PUBLIC LLAMAEDGE_ENABLE_KV_LAYER_EXPORT)
endif()

# EdgeWeaver per-layer KV import helper
option(LLAMAEDGE_ENABLE_KV_LAYER_IMPORT "llama: enable per-layer KV import helper" OFF)
if(LLAMAEDGE_ENABLE_KV_LAYER_IMPORT)
    target_compile_definitions(llama PUBLIC LLAMAEDGE_ENABLE_KV_LAYER_IMPORT)
endif()

if(LLAMAEDGE_ENABLE_KV_LAYER_EXPORT OR LLAMAEDGE_ENABLE_KV_LAYER_IMPORT)
    target_sources(llama PRIVATE
        llama-edgeweaver-kv.cpp
        llama-edgeweaver-kv-api.cpp)
endif()

option(LLAMAEDGE_ENABLE_SPLIT_GRAPH "llama: enable EdgeWeaver split-graph decode helper" OFF)
if(LLAMAEDGE_ENABLE_SPLIT_GRAPH)
    target_compile_definitions(llama PUBLIC LLAMAEDGE_ENABLE_SPLIT_GRAPH)
    target_sources(llama PRIVATE llama-edgeweaver-split.cpp)
endif()

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
