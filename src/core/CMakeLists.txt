add_library(ninfer_core STATIC
  arena.cu
  device.cu
  device_memory.cpp
  dtype.cpp
  host_worker_pool.cpp
  gdn_replay_records.cpp
  cyclic_kv_cache.cpp
  host_context_arena.cpp
  host_kv_arena.cpp
  paged_kv_cache.cpp
  linear_attention_state.cpp
  layout.cpp
  tensor.cpp
  weight_view.cpp
  decode_graph.cpp
)
ninfer_internal_includes(ninfer_core)
ninfer_cuda_archive(ninfer_core)
target_link_libraries(ninfer_core PUBLIC CUDA::cudart CUDA::nvtx3 Threads::Threads)
# NVML is loaded at run time (device_memory.cpp), so a machine without it falls back to CUDA.
target_link_libraries(ninfer_core PRIVATE ${CMAKE_DL_LIBS})
