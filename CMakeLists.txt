cmake_minimum_required(VERSION 3.18)
project(instantmesh_ggml VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ─────────────────────────────────────────────────────────────────────────
# Backend selection. ggml is vendored at third_party/ggml (v0.18.0). CUDA and
# Vulkan are *build-time* options of ggml; the inference code below is backend
# agnostic — at runtime it queries the ggml backend registry and picks the first
# GPU device (CUDA or Vulkan), falling back to CPU. Building both is achieved by
# simply enabling both options:
#
#   cmake -B build -DGGML_CUDA=ON -DGGML_VULKAN=ON
#
# We forward-control them here so they are defined before add_subdirectory.
# ─────────────────────────────────────────────────────────────────────────
option(GGML_CUDA   "Build ggml with the CUDA backend"   OFF)
option(GGML_VULKAN "Build ggml with the Vulkan backend" OFF)

add_subdirectory(third_party/ggml)
set(INSTANTMESH_GGML_TARGET ggml)

# ─────────────────────────────────────────────────────────────────────────
# Core library: backend selection + GGUF IO + tensor plumbing.
# ─────────────────────────────────────────────────────────────────────────
add_library(instantmesh_core STATIC
    src/core/backend.cpp
    src/core/gguf_io.cpp
    src/models/dino.cpp
    src/models/lrm_transformer.cpp
    src/models/synthesizer.cpp
    src/models/flexicubes.cpp
    src/models/texture_map.cpp
    src/models/xatlas/xatlas.cpp)
target_include_directories(instantmesh_core PUBLIC src)
target_link_libraries(instantmesh_core PUBLIC ${INSTANTMESH_GGML_TARGET})
target_compile_features(instantmesh_core PUBLIC cxx_std_17)
find_package(Threads REQUIRED)
target_link_libraries(instantmesh_core PRIVATE Threads::Threads)
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(instantmesh_core PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(instantmesh_core PRIVATE IM_USE_OPENMP=1)
endif()

# ─────────────────────────────────────────────────────────────────────────
# CLI tools (example programs; images live in examples/)
# ─────────────────────────────────────────────────────────────────────────
add_executable(instantmesh src/tools/instantmesh.cpp)
target_link_libraries(instantmesh PRIVATE instantmesh_core)

add_executable(dino src/tools/dino.cpp)
target_link_libraries(dino PRIVATE instantmesh_core)

add_executable(lrm_transformer src/tools/lrm_transformer.cpp)
target_link_libraries(lrm_transformer PRIVATE instantmesh_core)

add_executable(synthesizer src/tools/synthesizer.cpp)
target_link_libraries(synthesizer PRIVATE instantmesh_core)

add_executable(flexicubes src/tools/flexicubes.cpp)
target_link_libraries(flexicubes PRIVATE instantmesh_core)

# ─────────────────────────────────────────────────────────────────────────
# Parity tests (see docs/PLAN.md §3). Gate on fixtures being present.
# ─────────────────────────────────────────────────────────────────────────
option(INSTANTMESH_BUILD_TESTS "Build parity regression tests" ON)
if(INSTANTMESH_BUILD_TESTS)
    enable_testing()
    add_executable(test_backend tests/test_backend.cpp)
    target_link_libraries(test_backend PRIVATE instantmesh_core)
    add_test(NAME test_backend COMMAND test_backend)

    add_executable(test_flashattn tests/test_flashattn.cpp)
    target_link_libraries(test_flashattn PRIVATE instantmesh_core)
    add_test(NAME test_flashattn COMMAND test_flashattn)

    add_executable(test_mulmat_tiny tests/test_mulmat_tiny.cpp)
    target_link_libraries(test_mulmat_tiny PRIVATE instantmesh_core)
    add_test(NAME test_mulmat_tiny COMMAND test_mulmat_tiny)

    add_executable(test_cont_permute tests/test_cont_permute.cpp)
    target_link_libraries(test_cont_permute PRIVATE instantmesh_core)
    add_test(NAME test_cont_permute COMMAND test_cont_permute)

    add_executable(test_texture_map tests/test_texture_map.cpp)
    target_link_libraries(test_texture_map PRIVATE instantmesh_core)
    add_test(NAME test_texture_map COMMAND test_texture_map)
endif()