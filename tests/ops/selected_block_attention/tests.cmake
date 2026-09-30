# selected_block_attention: Op qualification tests.
ninfer_add_op_test(ninfer_selected_block_attention_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_selected_block_attention.cpp"
  LIBRARIES ninfer_ops)
