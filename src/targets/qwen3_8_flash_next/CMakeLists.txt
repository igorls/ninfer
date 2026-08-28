add_library(ninfer_qwen3_8_flash_next STATIC
  impl/expert_bank.cpp
  impl/load/bindings.cpp
  impl/load/materialized.cpp
  impl/ple_index.cpp
  impl/ple_table.cpp)
ninfer_internal_includes(ninfer_qwen3_8_flash_next)
target_include_directories(ninfer_qwen3_8_flash_next PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ninfer_qwen3_8_flash_next PUBLIC ninfer_artifact)
