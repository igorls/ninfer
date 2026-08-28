add_library(ninfer_qwen3_8_flash_next STATIC
  impl/expert_bank.cpp
  impl/gdn.cpp
  impl/gdn_kernels.cu
  impl/hyper_connection.cpp
  impl/hyper_connection_kernels.cu
  impl/load/bindings.cpp
  impl/load/materialized.cpp
  impl/moe.cpp
  impl/moe_kernels.cu
  impl/moe_route.cu
  impl/ple_decode.cpp
  impl/ple_decode_kernels.cu
  impl/ple_index.cpp
  impl/ple_pipeline.cpp
  impl/ple_table.cpp
  impl/qsa_attention.cpp
  impl/qsa_attention_kernels.cu
  impl/qsa_indexer.cpp
  impl/qsa_indexer_kernels.cu
  impl/text_decode.cpp
  impl/text_decode_kernels.cu)
ninfer_internal_includes(ninfer_qwen3_8_flash_next)
target_include_directories(ninfer_qwen3_8_flash_next PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ninfer_qwen3_8_flash_next PUBLIC ninfer_artifact ninfer_ops)
ninfer_cuda_archive(ninfer_qwen3_8_flash_next)
