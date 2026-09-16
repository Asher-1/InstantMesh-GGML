cmake_minimum_required(VERSION 3.18)
project(instantmesh_ggml VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ─────────────────────────────────────────────────────────────────────────
# Backend selection. ggml is vendored at third_party/ggml (submodule,
# upstream v0.18.1 + patches/*.patch, currently the RMBG custom ops).
# CUDA and
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

# ─────────────────────────────────────────────────────────────────────────
# ggml submodule guard + adaptation patches.
#
# Any local modification to the vendored ggml MUST live as a .patch file in
# patches/ (see patches/README.md) — submodule working-tree changes are NOT
# distributed with this repo, so patches are applied automatically here to
# keep `clone -> submodule update -> build` reproducible for everyone.
# The patch set is currently EMPTY (ggml is pristine upstream v0.18.1).
# ─────────────────────────────────────────────────────────────────────────
set(INSTANTMESH_GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml")
if(NOT EXISTS "${INSTANTMESH_GGML_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "ggml submodule not found at ${INSTANTMESH_GGML_DIR}\n"
        "Run:   git submodule update --init --recursive")
endif()

file(GLOB INSTANTMESH_GGML_PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch")
list(SORT INSTANTMESH_GGML_PATCHES)
if(INSTANTMESH_GGML_PATCHES)
    find_package(Git REQUIRED)
    foreach(_patch IN LISTS INSTANTMESH_GGML_PATCHES)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${INSTANTMESH_GGML_DIR}" apply --check "${_patch}"
            RESULT_VARIABLE _patch_check
            OUTPUT_QUIET ERROR_QUIET)
        if(_patch_check EQUAL 0)
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" -C "${INSTANTMESH_GGML_DIR}" apply "${_patch}"
                RESULT_VARIABLE _patch_apply)
            if(NOT _patch_apply EQUAL 0)
                message(FATAL_ERROR "Failed to apply ggml patch: ${_patch}")
            endif()
            message(STATUS "Applied ggml patch: ${_patch}")
        else()
            # --check failed: either already applied (idempotent re-configure) or a
            # real conflict. `git apply --reverse --check` distinguishes the two.
            execute_process(
                COMMAND "${GIT_EXECUTABLE}" -C "${INSTANTMESH_GGML_DIR}" apply --reverse --check "${_patch}"
                RESULT_VARIABLE _patch_recheck
                OUTPUT_QUIET ERROR_QUIET)
            if(_patch_recheck EQUAL 0)
                message(STATUS "ggml patch already applied: ${_patch}")
            else()
                message(FATAL_ERROR
                    "ggml patch neither applies nor is applied (conflict?): ${_patch}\n"
                    "Regenerate it against the pinned ggml revision.")
            endif()
        endif()
    endforeach()
endif()

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
    src/models/xatlas/xatlas.cpp
    src/models/scheduler.cpp
    src/models/clip_vision.cpp
    src/models/vae.cpp
    src/models/unet.cpp)
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
# RMBG-2.0 (BiRefNet) background removal, vendored at
# third_party/RMBG-2.0-GGML. Built directly against the patched ggml target
# above on purpose — the submodule's own CMake materializes a second ggml
# copy, which we deliberately bypass. stb_image headers come from src/utils.
# ─────────────────────────────────────────────────────────────────────────
set(RMBG_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/RMBG-2.0-GGML")
add_library(rmbg_core STATIC
    ${RMBG_ROOT}/src/backend.cpp
    ${RMBG_ROOT}/src/birefnet_decoder.cpp
    ${RMBG_ROOT}/src/birefnet_encoder.cpp
    ${RMBG_ROOT}/src/engine.cpp
    ${RMBG_ROOT}/src/model_loader.cpp
    ${RMBG_ROOT}/src/nn_ops.cpp
    ${RMBG_ROOT}/src/preprocess.cpp
    ${RMBG_ROOT}/src/rmbg_capi.cpp
    ${RMBG_ROOT}/src/rmbg_graph.cpp
    ${RMBG_ROOT}/src/swin_backbone.cpp
    ${RMBG_ROOT}/src/swin_graph.cpp)
target_include_directories(rmbg_core PUBLIC
    ${RMBG_ROOT}/include
    ${RMBG_ROOT}/src
    src/utils)
target_link_libraries(rmbg_core PUBLIC ${INSTANTMESH_GGML_TARGET})
target_compile_features(rmbg_core PUBLIC cxx_std_17)

add_executable(rembg src/tools/rembg.cpp)
target_link_libraries(rembg PRIVATE rmbg_core)

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

    add_executable(test_conv_layout tests/test_conv_layout.cpp)
    target_link_libraries(test_conv_layout PRIVATE instantmesh_core)
    add_test(NAME test_conv_layout COMMAND test_conv_layout)

    add_executable(test_texture_map tests/test_texture_map.cpp)
    target_link_libraries(test_texture_map PRIVATE instantmesh_core)
    add_test(NAME test_texture_map COMMAND test_texture_map)

    add_executable(test_scheduler tests/test_scheduler.cpp)
    target_link_libraries(test_scheduler PRIVATE instantmesh_core)
    target_compile_definitions(test_scheduler PRIVATE
        SCHEDULER_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/fixtures/scheduler")
    add_test(NAME test_scheduler COMMAND test_scheduler
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(test_clip_vision tests/test_clip_vision.cpp)
    target_link_libraries(test_clip_vision PRIVATE instantmesh_core)
    target_compile_definitions(test_clip_vision PRIVATE
        CLIPV_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/fixtures/clip_vision"
        CLIPV_MODEL="${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/zero123pp_cond_f32.gguf")
    add_test(NAME test_clip_vision COMMAND test_clip_vision
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    add_executable(test_vae tests/test_vae.cpp)
    target_link_libraries(test_vae PRIVATE instantmesh_core)
    target_compile_definitions(test_vae PRIVATE
        VAE_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/fixtures/vae"
        VAE_MODEL="${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/zero123pp_vae_f32.gguf")
    add_test(NAME test_vae COMMAND test_vae
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_tests_properties(test_vae PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(test_unet tests/test_unet.cpp)
    target_link_libraries(test_unet PRIVATE instantmesh_core)
    target_compile_definitions(test_unet PRIVATE
        UNET_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/fixtures/unet"
        UNET_MODEL="${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/zero123pp_unet_f32.gguf")
    add_test(NAME test_unet COMMAND test_unet
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_tests_properties(test_unet PROPERTIES SKIP_RETURN_CODE 77)
endif()