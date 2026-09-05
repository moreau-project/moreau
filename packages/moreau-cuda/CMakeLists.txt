cmake_minimum_required(VERSION 3.22)

# CMP0135: set timestamps on extracted files to extraction time (CMake 3.24+)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# Set CUDA architecture BEFORE project() so cmake doesn't override it.
# 75: Turing (T4), 80: Ampere (A100), 86: Ampere (A10G), 89: Ada (L4), 90: Hopper (H100)
# 120: Blackwell (RTX 5090)
#
# Override with cmake:  cmake .. -DCMAKE_CUDA_ARCHITECTURES=120
# Override with env:    MOREAU_CUDA_ARCH=120 cmake ..
# Override for pip:     MOREAU_CUDA_ARCH=120 pip install -e . --no-build-isolation
# Default: auto-detect via __nvcc_device_query, falls back to multi-arch.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(DEFINED ENV{MOREAU_CUDA_ARCH})
        set(CMAKE_CUDA_ARCHITECTURES "$ENV{MOREAU_CUDA_ARCH}")
    else()
        # Try runtime detection via __nvcc_device_query (more reliable than cmake native)
        execute_process(
            COMMAND __nvcc_device_query
            OUTPUT_VARIABLE _detected_arch
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _detect_result
        )
        if(_detect_result EQUAL 0 AND _detected_arch MATCHES "^[0-9]+$")
            set(CMAKE_CUDA_ARCHITECTURES "${_detected_arch}")
            message(STATUS "Auto-detected CUDA architecture: sm_${_detected_arch}")
        else()
            # Only include architectures supported by the toolkit.
            # sm_120 (Blackwell) requires CUDA 12.8+; detect via nvcc --list-gpu-arch.
            find_program(_nvcc_path nvcc HINTS ENV CUDA_PATH PATH_SUFFIXES bin)
            if(_nvcc_path)
                execute_process(
                    COMMAND ${_nvcc_path} --list-gpu-arch
                    OUTPUT_VARIABLE _supported_archs
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET
                    RESULT_VARIABLE _list_result
                )
            else()
                set(_list_result 1)
            endif()
            set(_fallback_archs "75;80;86;89;90")
            if(_list_result EQUAL 0 AND _supported_archs MATCHES "compute_120")
                list(APPEND _fallback_archs "120")
            endif()
            set(CMAKE_CUDA_ARCHITECTURES "${_fallback_archs}")
        endif()
    endif()
endif()

project(moreau-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# Find cuDSS - check standard paths and CUDSS_ROOT
if(DEFINED ENV{CUDSS_ROOT})
    set(CUDSS_ROOT $ENV{CUDSS_ROOT})
endif()

# Add cuDSS include/lib paths
if(CUDSS_ROOT)
    include_directories(${CUDSS_ROOT}/include)
    link_directories(${CUDSS_ROOT}/lib)
endif()

# Also check /usr/local (common install location)
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Moreau CUDA library
add_library(moreau_cuda_lib STATIC
    src/equilibration/equilibration.cpp
    src/equilibration/equilibration_kernels.cu
    src/kkt/kkt.cpp
    src/kkt/kkt_solver.cpp
    src/kkt/kkt_kernels.cu
    src/kkt/riccati.cpp
    src/kkt/riccati_kernels.cu
    src/kkt/kkt_woodbury.cpp
    src/kkt/woodbury_kernels.cu
    src/kkt/sqmr.cu
    src/cones/cones.cpp
    src/cones/cones.cu
    src/cones/barrier.cu
    src/cones/soc_kernels.cu
    src/cones/composite_kernels.cu
    src/cones/genpow_pd_kernels.cu
    src/cones/psd_kernels.cu
    src/cones/smoothing_kernels.cu
    src/solver/warm_start_kernels.cu
    src/vector/vector_kernels.cu
    src/residuals/residuals_kernels.cu
    src/residuals/residuals.cpp
    src/variables/variables.cpp
    src/variables/variables_kernels.cu
    src/solution/solution.cpp
    src/solver/solver.cpp
    src/solver/info.cpp
    src/solver/solver_kernels.cu
    src/diff/diff.cu
    src/diff/diff_kernels.cu
    src/diff/smoothed_kernels.cu
    src/diff/diff_psd.cu
    src/diff/diff_kkt.cu
    src/diff/diff_woodbury.cu
    src/diff/diff_riccati.cu
    src/chordal/chordal_info.cpp
)
target_include_directories(moreau_cuda_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(moreau_cuda_lib PUBLIC cudss CUDA::cudart CUDA::cusparse CUDA::cublas CUDA::cusolver)
set_property(TARGET moreau_cuda_lib PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET moreau_cuda_lib PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

# Hide all symbols by default so internal implementation details are not exported
# from shared libraries that link this static library
set_target_properties(moreau_cuda_lib PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
target_compile_options(moreau_cuda_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility=hidden>
    # Strip source paths from binaries to avoid leaking build environment info
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=.>
    $<$<COMPILE_LANGUAGE:CXX>:-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=.>
    # No debug info in release builds
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-g0>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-g0>
)

# Opt-in profiling: -DMOREAU_ENABLE_PROFILING=ON defines the macro of the same
# name, which gates per-phase cudaEvent timing accumulators and an env-gated
# stderr dump at end of solve (runtime gated by MOREAU_PROFILE_ENABLED=1).
if(MOREAU_ENABLE_PROFILING)
    target_compile_definitions(moreau_cuda_lib PUBLIC MOREAU_ENABLE_PROFILING)
endif()


# =============================================================================
# C API version (read from pyproject.toml, single source of truth)
# =============================================================================
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _pyproject_version_line
     REGEX "^version = ")
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" MOREAU_VERSION "${_pyproject_version_line}")
if(NOT MOREAU_VERSION)
    message(FATAL_ERROR "Could not parse version from pyproject.toml")
endif()
string(REGEX MATCH "^[0-9]+" MOREAU_VERSION_MAJOR "${MOREAU_VERSION}")

# =============================================================================
# C API static library
# =============================================================================
add_library(moreau_cuda_c_api STATIC src/c_api/c_api.cpp)
target_link_libraries(moreau_cuda_c_api PUBLIC moreau_cuda_lib)
target_include_directories(moreau_cuda_c_api PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../moreau-c/include
)
target_compile_definitions(moreau_cuda_c_api PRIVATE MOREAU_VERSION="${MOREAU_VERSION}")
set_property(TARGET moreau_cuda_c_api PROPERTY POSITION_INDEPENDENT_CODE ON)

# =============================================================================
# C API shared library (for non-Python consumers: Julia, C, etc.)
# =============================================================================
option(MOREAU_BUILD_C_SHARED "Build shared library for C API" OFF)

if(MOREAU_BUILD_C_SHARED)

    add_library(moreau_cuda_shared SHARED src/c_api/c_api.cpp)
    target_link_libraries(moreau_cuda_shared PRIVATE moreau_cuda_lib)
    target_include_directories(moreau_cuda_shared PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../moreau-c/include
    )
    target_compile_definitions(moreau_cuda_shared PRIVATE MOREAU_VERSION="${MOREAU_VERSION}")
    set_target_properties(moreau_cuda_shared PROPERTIES
        OUTPUT_NAME moreau_cuda
        # Export only moreau_* symbols via version script
        LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/moreau_c_api_exports.lds"
        # SONAME for proper dynamic linking
        SOVERSION ${MOREAU_VERSION_MAJOR}
        VERSION ${MOREAU_VERSION}
        BUILD_RPATH ""
        INSTALL_RPATH ""
        SKIP_BUILD_RPATH TRUE
        BUILD_WITH_INSTALL_RPATH FALSE
    )
    # Strip in release builds
    if(CMAKE_BUILD_TYPE STREQUAL "Release" AND CMAKE_STRIP)
        add_custom_command(TARGET moreau_cuda_shared POST_BUILD
            COMMAND ${CMAKE_STRIP} --strip-unneeded $<TARGET_FILE:moreau_cuda_shared>
            COMMENT "Stripping C API shared library"
        )
    endif()
    message(STATUS "C API shared library enabled (libmoreau_cuda.so)")
endif()

# =============================================================================
# Python bindings (optional - requires Python and nanobind)
# =============================================================================
option(MOREAU_BUILD_PYTHON "Build Python bindings" ON)

if(MOREAU_BUILD_PYTHON)
    # Prefer venv Python if VIRTUAL_ENV is set and Python_EXECUTABLE not specified
    if(NOT Python_EXECUTABLE AND DEFINED ENV{VIRTUAL_ENV})
        set(Python_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python3")
    endif()

    find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule QUIET)
    if(Python_FOUND)
        # Find nanobind - get cmake dir from Python
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c "import nanobind; print(nanobind.cmake_dir())"
            OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE nanobind_result
        )

        if(nanobind_result EQUAL 0)
            list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
            find_package(nanobind CONFIG REQUIRED)

            nanobind_add_module(
                _moreau_cuda
                STABLE_ABI
                NB_STATIC
                bindings/moreau_bindings.cpp
            )
            target_link_libraries(_moreau_cuda PRIVATE moreau_cuda_lib)
            target_include_directories(_moreau_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

            # Strip source paths and debug info from Python binding
            target_compile_options(_moreau_cuda PRIVATE
                -ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=.
                $<$<CONFIG:Release>:-g0>
            )

            if(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
                set_target_properties(_moreau_cuda PROPERTIES
                    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
                )
            else()
                set_target_properties(_moreau_cuda PROPERTIES
                    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/moreau_cuda"
                )
            endif()
            install(TARGETS _moreau_cuda DESTINATION moreau_cuda)
            message(STATUS "Python bindings enabled (nanobind with stable ABI)")
        else()
            message(STATUS "nanobind not found - Python bindings will not be built")
        endif()

        # JAX FFI bindings (optional - requires JAX with XLA FFI headers)
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c "from jax._src import ffi; print(ffi.include_dir())"
            OUTPUT_VARIABLE JAX_FFI_INCLUDE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE jax_ffi_result
        )
        if(jax_ffi_result EQUAL 0 AND EXISTS "${JAX_FFI_INCLUDE_DIR}")
            add_library(_moreau_jax_ffi SHARED bindings/jax_ffi.cu)
            target_link_libraries(_moreau_jax_ffi PRIVATE moreau_cuda_lib)
            target_include_directories(_moreau_jax_ffi PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${JAX_FFI_INCLUDE_DIR}
            )
            set_property(TARGET _moreau_jax_ffi PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

            # Strip source paths and debug info from JAX FFI binary
            target_compile_options(_moreau_jax_ffi PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}=.>
                $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-g0>
            )

            # Export only the three FFI entry points via version script.
            # Internal moreau symbols are already hidden (moreau_cuda_lib is compiled
            # with -fvisibility=hidden). The version script controls which symbols
            # from jax_ffi.cu itself are exported.
            target_link_options(_moreau_jax_ffi PRIVATE
                "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/bindings/jax_ffi_exports.lds"
            )

            # Strip the JAX FFI library in Release builds
            if(CMAKE_BUILD_TYPE STREQUAL "Release" AND CMAKE_STRIP)
                add_custom_command(TARGET _moreau_jax_ffi POST_BUILD
                    COMMAND ${CMAKE_STRIP} --strip-unneeded $<TARGET_FILE:_moreau_jax_ffi>
                    COMMENT "Stripping JAX FFI library"
                )
            endif()

            if(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
                set_target_properties(_moreau_jax_ffi PROPERTIES
                    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
                )
            else()
                set_target_properties(_moreau_jax_ffi PROPERTIES
                    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/moreau_cuda"
                )
            endif()
            message(STATUS "JAX FFI bindings enabled (include: ${JAX_FFI_INCLUDE_DIR})")
        else()
            message(STATUS "JAX FFI headers not found - JAX zero-copy bindings will not be built")
        endif()
    else()
        message(STATUS "Python not found - Python bindings will not be built")
    endif()
endif()

# =============================================================================
# Tests (require Google Test)
# =============================================================================
option(MOREAU_BUILD_TESTS "Build C++ tests" ON)

if(MOREAU_BUILD_TESTS)
    find_package(GTest QUIET)
    if(GTest_FOUND)
        enable_testing()

        # Test executables
        set(TEST_SOURCES
            tests/cpp/test_batch.cpp
            tests/cpp/test_batch_b_vectors.cpp
            tests/cpp/test_batch_scaling_bug.cpp
            tests/cpp/test_batch_consistency.cpp
            tests/cpp/test_clarabel_ported.cpp
            tests/cpp/test_cone_varieties.cpp
            tests/cpp/test_convergence.cpp
            tests/cpp/test_cvxpy_ported.cpp
            tests/cpp/test_data_updating.cpp
            tests/cpp/test_diff.cpp
            tests/cpp/test_dimension_checks.cpp
            tests/cpp/test_empty_A_matrix.cpp
            tests/cpp/test_eq_constrained.cpp
            tests/cpp/test_equilibration.cpp
            tests/cpp/test_exp.cpp
            tests/cpp/test_genpowcone.cpp
            tests/cpp/test_inf_in_b.cpp
            tests/cpp/test_infeasibility.cpp
            tests/cpp/test_kkt.cpp
            tests/cpp/test_kkt_integration.cpp
            tests/cpp/test_lp.cpp
            tests/cpp/test_matrix.cpp
            tests/cpp/test_mixed_conic.cpp
            tests/cpp/test_numerical_stability.cpp
            tests/cpp/test_objective_values.cpp
            tests/cpp/test_pnorm.cpp
            tests/cpp/test_power.cpp
            tests/cpp/test_psd.cpp
            tests/cpp/test_psd_chordal.cpp
            tests/cpp/test_psd_fuzz.cpp
            tests/cpp/test_psd_scaling_flag.cpp
            tests/cpp/test_qp.cpp
            tests/cpp/test_random_problems.cpp
            tests/cpp/test_settings.cpp
            tests/cpp/test_soc.cpp
            tests/cpp/test_soc_cudss_debug.cpp
            tests/cpp/test_soc_hard.cpp
            tests/cpp/test_soc_kkt_dump.cpp
            tests/cpp/test_soc_large_dim_stress.cpp
            tests/cpp/test_soc_varlen.cpp
            tests/cpp/test_solution_quality.cpp
            tests/cpp/test_solver.cpp
            tests/cpp/test_solver_multicone.cpp
            tests/cpp/test_subtle_edge_cases.cpp
            tests/cpp/test_termination.cpp
            tests/cpp/test_best_iterate.cpp
            tests/cpp/test_yolo.cpp
            tests/cpp/test_unconstrained.cpp
            tests/cpp/test_vector.cpp
            tests/cpp/test_riccati.cpp
            tests/cpp/test_woodbury.cpp
            tests/cpp/test_warm_start.cpp
            tests/cpp/test_direct_cone_scaffolding.cpp
            tests/cpp/test_direct_cone_endtoend.cpp
            tests/cpp/test_direct_cone_infeasibility.cpp
            tests/cpp/test_direct_cone_backward.cpp
            tests/cpp/test_direct_cone_exp.cpp
            tests/cpp/test_direct_cone_pow.cpp
            tests/cpp/test_direct_cone_genpow.cpp
            tests/cpp/test_direct_cone_asymm_backward.cpp
            tests/cpp/test_multi_psd_mre.cpp
            tests/cpp/test_compiled_solver_move_semantics.cpp
        )

        foreach(test_source ${TEST_SOURCES})
            get_filename_component(test_name ${test_source} NAME_WE)
            add_executable(${test_name} ${test_source})
            target_link_libraries(${test_name} PRIVATE moreau_cuda_lib GTest::gtest GTest::gtest_main)
            target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
            add_test(NAME ${test_name} COMMAND ${test_name})
        endforeach()

        # C API test
        add_executable(test_c_api tests/cpp/test_c_api.cpp)
        target_link_libraries(test_c_api PRIVATE moreau_cuda_c_api GTest::gtest GTest::gtest_main)
        target_include_directories(test_c_api PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../moreau-c/include
        )
        add_test(NAME test_c_api COMMAND test_c_api)

        message(STATUS "Tests enabled - ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp")
    else()
        message(STATUS "GTest not found - tests will not be built")
    endif()
endif()

# =============================================================================
# Examples
# =============================================================================
option(MOREAU_BUILD_EXAMPLES "Build C++ examples" ON)

if(MOREAU_BUILD_EXAMPLES)
    # Examples that do NOT require GTest (standalone)
    set(EXAMPLE_SOURCES
        examples/cpp/solve_multicone.cpp
        examples/cpp/benchmark_mpc_riccati.cpp
        examples/cpp/large_qp_batched.cpp
    )

    foreach(example_source ${EXAMPLE_SOURCES})
        get_filename_component(example_name ${example_source} NAME_WE)
        add_executable(${example_name} ${example_source})
        target_link_libraries(${example_name} PRIVATE moreau_cuda_lib)
        target_include_directories(${example_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    endforeach()

    # Examples that require GTest (simple_qp is a gtest-based example)
    if(GTest_FOUND)
        add_executable(simple_qp examples/cpp/simple_qp.cpp)
        target_link_libraries(simple_qp PRIVATE moreau_cuda_lib GTest::gtest GTest::gtest_main)
        target_include_directories(simple_qp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    endif()

    # Standalone benchmarks (no GTest)
    add_executable(bench_yolo examples/cpp/bench_yolo.cpp)
    target_link_libraries(bench_yolo PRIVATE moreau_cuda_lib)
    target_include_directories(bench_yolo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    add_executable(bench_soc_scaling examples/cpp/bench_soc_scaling.cpp)
    target_link_libraries(bench_soc_scaling PRIVATE moreau_cuda_lib)
    target_include_directories(bench_soc_scaling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    add_executable(bench_direct_cone_asymmetric examples/cpp/bench_direct_cone_asymmetric.cpp)
    target_link_libraries(bench_direct_cone_asymmetric PRIVATE moreau_cuda_lib)
    target_include_directories(bench_direct_cone_asymmetric PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    add_executable(bench_eta_perf ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/bench_eta_perf.cpp)
    target_link_libraries(bench_eta_perf PRIVATE moreau_cuda_lib)
    target_include_directories(bench_eta_perf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    add_executable(bench_direct_cone_backward ${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/bench_direct_cone_backward.cpp)
    target_link_libraries(bench_direct_cone_backward PRIVATE moreau_cuda_lib)
    target_include_directories(bench_direct_cone_backward PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    message(STATUS "Examples enabled - ${CMAKE_CURRENT_SOURCE_DIR}/examples/cpp")
endif()
