target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/execute.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/input_conv.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/convolution_decode.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/convolution_window.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/projection_a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/projection_a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/projection_decode.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/projection_window.cu"
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/projected_conv.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/convolution.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/projection.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/projection_independent.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/projection_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/projection.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q8/projection_decode.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/projection_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/sliced_k.cu"
)

target_sources(ninfer_ops_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/projection_a4_tma.cu"
)

target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_conv_fused.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_input_a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_input_decode.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_input_matrix.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_input_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_input_k2560_fp32.cu"
)
include("${CMAKE_CURRENT_LIST_DIR}/${NINFER_TARGET}/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/q8/${NINFER_TARGET}/sources.cmake")
