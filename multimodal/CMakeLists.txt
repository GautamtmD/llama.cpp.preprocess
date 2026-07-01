# multimodal-server — MultiModalAgent C++ LLM server.
#
# This is a TOP-LEVEL CMake project. It pulls in the llama.cpp fork (the parent
# directory of this file) via add_subdirectory so we can link `llama`,
# `llama-common`, and the vendored `cpp-httplib` WITHOUT modifying the submodule.
# All our code stays under engine/multimodal/.

cmake_minimum_required(VERSION 3.14)
project(multimodal-server CXX C)

# Pull in the upstream llama.cpp fork. EXCLUDE_FROM_ALL so its own targets only
# build if something depends on them (we depend on llama/llama-common/cpp-httplib).
set(LLAMA_STANDALONE OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
# We DO need the `common` lib (llama-common) for tokenization helpers etc.
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_BINARY_DIR}/llama-build
    EXCLUDE_FROM_ALL
)

add_executable(multimodal-server
    src/main.cpp
)

# Put the exe next to the fork's runtime DLLs (llama.dll, ggml.dll, ...) so they
# resolve at launch without PATH tricks; otherwise ggml_backend_load_all() can't
# find the backend DLLs and model load fails with "no backends are loaded".
# For multi-config generators (VS), $<CONFIG> avoids the double "Release/Release".
set_target_properties(multimodal-server PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>
)

target_include_directories(multimodal-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/cpp-httplib
    ${CMAKE_CURRENT_SOURCE_DIR}/../vendor
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(multimodal-server PRIVATE
    llama
    llama-common
    cpp-httplib
)

target_compile_features(multimodal-server PRIVATE cxx_std_17)
