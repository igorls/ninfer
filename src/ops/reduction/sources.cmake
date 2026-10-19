target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/argmax.cu"
  "${CMAKE_CURRENT_LIST_DIR}/target_logprobs.cpp"
)

target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/target_logprobs_launch.cu" "${CMAKE_CURRENT_LIST_DIR}/candidate_logprobs.cpp" "${CMAKE_CURRENT_LIST_DIR}/candidate_logprobs_launch.cu")
