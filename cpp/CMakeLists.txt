cmake_minimum_required(VERSION 3.16)
project(ELektron2_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Boost (header-only for odeint)
find_package(Boost REQUIRED)

# OpenMP
find_package(OpenMP)

add_executable(elektron2 src/main.cpp)

target_include_directories(elektron2 PRIVATE
    ${Boost_INCLUDE_DIRS}
    src/
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(elektron2 PRIVATE OpenMP::OpenMP_CXX)
endif()
