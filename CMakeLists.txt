cmake_minimum_required(VERSION 3.28)

include(cmake/NinferPlatform.cmake)

project(ninfer LANGUAGES C CXX CUDA)
ninfer_validate_platform()

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_ENGINE "Build models, frontend and Engine" ON)
option(NINFER_BUILD_APPS "Build the NInfer CLI, server and perplexity evaluator" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)
option(NINFER_ENABLE_DISTRIBUTED "Build distributed execution and NCCL communication" OFF)

if(NINFER_BUILD_APPS AND NOT NINFER_BUILD_ENGINE)
  message(FATAL_ERROR "NINFER_BUILD_APPS requires NINFER_BUILD_ENGINE")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# NInfer has no C++20 modules; scanning makes MSVC run a dependency pass over every source.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS ninfer_link=1)
  set(CMAKE_JOB_POOL_LINK ninfer_link)
endif()

if(MSVC)
  # Sources are UTF-8; MSVC otherwise reads them in the system code page, which corrupts
  # non-ASCII string literals and fails fmt's UTF-8 static_assert.
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
  # CUDA 13's CCCL headers require MSVC's standard-conforming preprocessor.
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
  # windows.h defines min/max macros that break std::min/std::max.
  add_compile_definitions(NOMINMAX)
endif()

# Tests also exercise product input, logging and protocol adapters.
set(NINFER_BUILD_PRODUCT_SUPPORT OFF)
if(NINFER_BUILD_ENGINE AND (NINFER_BUILD_APPS OR BUILD_TESTING))
  set(NINFER_BUILD_PRODUCT_SUPPORT ON)
endif()

if(WIN32)
  set(NINFER_LIBRARY_TYPE STATIC)
else()
  set(NINFER_LIBRARY_TYPE SHARED)
endif()

include(cmake/Dependencies.cmake)
include(cmake/NinferTargets.cmake)

# Project binaries share the same libraries and CUDA runtime. These defaults are
# set after third-party targets so their build policies remain independent.
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_LINK_DEPENDS_NO_SHARED ON)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
