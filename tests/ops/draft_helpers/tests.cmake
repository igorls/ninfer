ninfer_add_op_test(ninfer_draft_normalization_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../draft_helpers/draft_normalization.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/../normalization/gathered_features.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/../normalization/rmsnorm_pack_tail.cpp"
  LIBRARIES ninfer_ops ninfer_op_suite)

ninfer_add_op_test(ninfer_draft_convolution_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../draft_helpers/draft_convolution.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/../dynamic_grouped_conv/prepare.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/../dynamic_grouped_conv/linear_finish.cpp"
  LIBRARIES ninfer_ops ninfer_op_suite)

ninfer_add_op_test(ninfer_draft_embedding_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../draft_helpers/draft_embedding.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/../embedding/embedding.cpp"
  LIBRARIES ninfer_ops ninfer_op_suite)

ninfer_add_op_test(ninfer_draft_rope_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../draft_helpers/draft_rope.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/../rope/rmsnorm_rope.cpp"
  LIBRARIES ninfer_ops ninfer_op_suite)

ninfer_add_op_test(ninfer_gathered_sampling_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../draft_helpers/gathered_sampling.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/../sampling/gathered_selector.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/../sampling/vocabulary_sparse.cpp"
  LIBRARIES ninfer_ops ninfer_op_suite)

ninfer_add_op_test(ninfer_draft_context_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../kv_cache/context_kv_materialize.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/../kv_cache/context_kv_materialize_bf16.cpp"
  LIBRARIES ninfer_ops ninfer_op_suite)

ninfer_add_op_test(ninfer_draft_sliding_attention_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../softmax_attention/sliding_window_attention.cpp"
  LIBRARIES ninfer_ops ninfer_op_suite)

if(NINFER_TARGET STREQUAL "dgx_spark")
  ninfer_add_op_test(ninfer_draft_topk_keys_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/topk_keys.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../linear_topk/keys.cpp"
            "${CMAKE_CURRENT_LIST_DIR}/../linear/support.cpp"
    LIBRARIES ninfer_ops ninfer_op_suite)
endif()
