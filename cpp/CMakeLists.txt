cmake_minimum_required(VERSION 3.16)
project(ELektron2_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# ---- Boost (header-only, always available) ----
find_package(Boost REQUIRED)

# ---- SFML (optional — PlotDots visualization) ----
find_package(SFML 2.5 COMPONENTS graphics window system QUIET)
if(SFML_FOUND)
    message(STATUS "SFML found — PlotDots visualization enabled")
else()
    message(STATUS "SFML not found — building without visualization (install libsfml-dev)")
endif()

# ---- OpenMP ----
find_package(OpenMP)

# ---- Sources ----
set(SOURCES src/main.cpp)

add_executable(elektron2 ${SOURCES})

target_include_directories(elektron2 PRIVATE src/)
target_link_libraries(elektron2 PRIVATE Boost::boost)

if(SFML_FOUND)
    find_package(X11 REQUIRED)
    target_compile_definitions(elektron2 PRIVATE HAVE_SFML)
    target_link_libraries(elektron2 PRIVATE sfml-graphics sfml-window sfml-system ${X11_LIBRARIES})
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(elektron2 PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---- CUDA target (optional — GPU-parallel DP853) ----
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Auto-detect GPU architecture
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()

    # FP32 build (fast — 64x more throughput on consumer GPUs)
    add_executable(elektron2_cuda src/cuda/elektron_cuda.cu)
    set_target_properties(elektron2_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    target_compile_options(elektron2_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O2 --use_fast_math -DUSE_FLOAT>
    )

    # FP64 build (precise — slow on consumer GPUs, fast on datacenter GPUs)
    add_executable(elektron2_cuda_fp64 src/cuda/elektron_cuda.cu)
    set_target_properties(elektron2_cuda_fp64 PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    target_compile_options(elektron2_cuda_fp64 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O2>
    )

    message(STATUS "CUDA found (${CMAKE_CUDA_COMPILER}) — building elektron2_cuda (FP32) + elektron2_cuda_fp64")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "CUDA not found — skipping elektron2_cuda target")
endif()

# ---- ROCm/HIP target (optional — AMD GPU-parallel DP853) ----
# HIP language support requires CMake >= 3.21; check_language(HIP) verifies hipcc exists.
find_package(hip QUIET)
if(hip_FOUND)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        enable_language(HIP)
        set(CMAKE_HIP_STANDARD 17)
        set(CMAKE_HIP_STANDARD_REQUIRED ON)

        # Mark the source as HIP so CMake uses hipcc instead of c++
        set_source_files_properties(src/rocm/elektron_rocm.cpp PROPERTIES LANGUAGE HIP)

        # FP32 build
        add_executable(elektron2_rocm src/rocm/elektron_rocm.cpp)
        target_link_libraries(elektron2_rocm PRIVATE hip::device)
        target_compile_options(elektron2_rocm PRIVATE
            $<$<COMPILE_LANGUAGE:HIP>:-DUSE_FLOAT -O2>
        )

        # FP64 build (AMD Instinct GPUs have full FP64 throughput — 1/2 of FP32)
        add_executable(elektron2_rocm_fp64 src/rocm/elektron_rocm.cpp)
        target_link_libraries(elektron2_rocm_fp64 PRIVATE hip::device)
        target_compile_options(elektron2_rocm_fp64 PRIVATE
            $<$<COMPILE_LANGUAGE:HIP>:-O2>
        )

        message(STATUS "ROCm/HIP found (${CMAKE_HIP_COMPILER}) — building elektron2_rocm (FP32) + elektron2_rocm_fp64")
    else()
        message(STATUS "ROCm/HIP package found but hipcc not available — skipping")
    endif()
else()
    message(STATUS "ROCm/HIP not found — skipping elektron2_rocm target")
endif()
