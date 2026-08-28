add_executable(ninfer_qwen3_8_flash_next_reference
  main.cpp
  options.cpp)
ninfer_internal_includes(ninfer_qwen3_8_flash_next_reference)
target_include_directories(ninfer_qwen3_8_flash_next_reference PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_8_flash_next
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
target_link_libraries(ninfer_qwen3_8_flash_next_reference PRIVATE
  ninfer_qwen3_8_flash_next
  ninfer_engine
  ninfer_artifact
  ninfer_core
  ninfer_ops
  CUDA::cudart)
