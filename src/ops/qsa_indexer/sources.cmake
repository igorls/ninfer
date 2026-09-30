# qsa_indexer: sources contributed to ninfer_ops.
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/qsa_indexer.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/launch.cu"
)
