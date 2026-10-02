add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
ninfer_internal_includes(ninfer)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
ninfer_internal_includes(ninfer-serve)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_logging)

if(WIN32)
  enable_language(RC)
  add_subdirectory(ninfer-supervisor)
  configure_file(windows/app.rc.in windows/app.rc @ONLY)
  foreach(app IN ITEMS ninfer ninfer-serve ninfer-supervisor)
    target_sources(${app} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/windows/app.rc")
  endforeach()
  set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)
  include(InstallRequiredSystemLibraries)
  get_filename_component(ninfer_linker_dir "${CMAKE_LINKER}" DIRECTORY)
  file(GENERATE OUTPUT "${PROJECT_BINARY_DIR}/windows-package-$<CONFIG>.cmake" CONTENT
"set(NINFER_EXECUTABLES [==[$<TARGET_FILE:ninfer-supervisor>;$<TARGET_FILE:ninfer-serve>;$<TARGET_FILE:ninfer>]==])
set(NINFER_RUNTIME_DIRS [==[${CUDAToolkit_BIN_DIR};${CUDAToolkit_BIN_DIR}/x64;${FFMPEG_ROOT}/bin;${CURL_ROOT}/bin]==])
set(NINFER_MSVC_RUNTIME [==[${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}]==])
set(CMAKE_GET_RUNTIME_DEPENDENCIES_PLATFORM windows+pe)
set(CMAKE_GET_RUNTIME_DEPENDENCIES_TOOL dumpbin)
set(CMAKE_GET_RUNTIME_DEPENDENCIES_COMMAND [==[${ninfer_linker_dir}/dumpbin.exe]==])
include([==[${PROJECT_SOURCE_DIR}/scripts/windows/package.cmake]==])
")
endif()

add_executable(ninfer-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp
  perplexity/reference.cpp)
ninfer_internal_includes(ninfer-perplexity)
target_link_libraries(ninfer-perplexity PRIVATE
  ninfer_engine ninfer_product_logging ninfer::json)

add_executable(ninfer-reasoning-collect reasoning-collect/main.cpp)
ninfer_internal_includes(ninfer-reasoning-collect)
target_link_libraries(ninfer-reasoning-collect PRIVATE ninfer_engine ninfer::json)

add_executable(ninfer-hidden-export hidden-export/main.cpp)
ninfer_internal_includes(ninfer-hidden-export)
target_link_libraries(ninfer-hidden-export PRIVATE ninfer_engine ninfer::json)
