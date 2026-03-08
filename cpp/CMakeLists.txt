cmake_minimum_required(VERSION 3.16)
project(ELektron2_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# -frounding-math needed for CAPD (harmless for Boost-only builds)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -DNDEBUG -frounding-math")
set(CMAKE_CXX_FLAGS_DEBUG "-g -frounding-math")

# ---- Boost (header-only, always available) ----
find_package(Boost REQUIRED)

# ---- CAPD (optional — only needed when Integrator::CAPD is selected) ----
set(CAPD_CONFIG "/home/marcf/capd-install/bin/capd-config" CACHE FILEPATH "Path to capd-config")

set(HAVE_CAPD OFF)
if(EXISTS "${CAPD_CONFIG}")
    execute_process(
        COMMAND ${CAPD_CONFIG} --cflags
        OUTPUT_VARIABLE CAPD_CFLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE CAPD_CFLAGS_RESULT
    )
    execute_process(
        COMMAND ${CAPD_CONFIG} --libs
        OUTPUT_VARIABLE CAPD_LIBS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE CAPD_LIBS_RESULT
    )
    if(CAPD_CFLAGS_RESULT EQUAL 0 AND CAPD_LIBS_RESULT EQUAL 0)
        set(HAVE_CAPD ON)
        message(STATUS "CAPD found via ${CAPD_CONFIG}")
        message(STATUS "CAPD cflags: ${CAPD_CFLAGS}")
        message(STATUS "CAPD libs:   ${CAPD_LIBS}")
    endif()
else()
    message(STATUS "CAPD not found (capd-config not at ${CAPD_CONFIG}) — Boost-only build")
endif()

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
if(HAVE_CAPD)
    list(APPEND SOURCES src/capd_logger_stub.cpp)
endif()

add_executable(elektron2 ${SOURCES})

target_include_directories(elektron2 PRIVATE src/)
target_link_libraries(elektron2 PRIVATE Boost::boost)

# Apply CAPD flags if available
if(HAVE_CAPD)
    separate_arguments(CAPD_CFLAGS_LIST UNIX_COMMAND "${CAPD_CFLAGS}")
    separate_arguments(CAPD_LIBS_LIST UNIX_COMMAND "${CAPD_LIBS}")
    target_compile_options(elektron2 PRIVATE ${CAPD_CFLAGS_LIST})
    target_link_libraries(elektron2 PRIVATE ${CAPD_LIBS_LIST})
endif()

if(SFML_FOUND)
    find_package(X11 REQUIRED)
    target_compile_definitions(elektron2 PRIVATE HAVE_SFML)
    target_link_libraries(elektron2 PRIVATE sfml-graphics sfml-window sfml-system ${X11_LIBRARIES})
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(elektron2 PRIVATE OpenMP::OpenMP_CXX)
endif()
