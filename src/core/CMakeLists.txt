if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR WIN32) OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  message(FATAL_ERROR "NInfer CPU execution requires Linux or Windows x86-64")
endif()

add_library(ninfer_cpu STATIC
  cpu/system.cpp
  cpu/task_pool.cpp
  cpu/executor.cpp
)
ninfer_internal_includes(ninfer_cpu)
target_link_libraries(ninfer_cpu PRIVATE ninfer::hwloc Threads::Threads)

if(WIN32)
  set(ninfer_io_sources io/file_windows.cpp io/read_queue_windows.cpp io/windows_calls.cpp)
else()
  set(ninfer_io_sources io/file.cpp io/read_queue.cpp)
endif()

add_library(ninfer_core STATIC
  arena.cpp
  buffer.cpp
  buffer.cu
  memory.cpp
  transfer.cpp
  transfer.cu
  device.cu
  device_memory.cpp
  dtype.cpp
  nf4.cpp
  u4z8.cpp
  ${ninfer_io_sources}
  gdn_replay_records.cpp
  cyclic_kv_cache.cpp
  host_context_arena.cpp
  host_kv_arena.cpp
  paged_kv_cache.cpp
  linear_attention_state.cpp
  state_image.cpp
  layout.cpp
  tensor.cpp
  weight_view.cpp
  weight_bank.cpp
  expert_cache.cpp
  decode_graph.cpp
)
ninfer_internal_includes(ninfer_core)
ninfer_cuda_archive(ninfer_core)
target_link_libraries(ninfer_core PUBLIC ninfer_cpu CUDA::cudart CUDA::nvtx3 Threads::Threads)
# NVML is loaded at run time (device_memory.cpp), so a machine without it falls back to CUDA.
target_link_libraries(ninfer_core PRIVATE ${CMAKE_DL_LIBS})
if(NOT WIN32)
  target_link_libraries(ninfer_core PRIVATE PkgConfig::URING)
endif()
