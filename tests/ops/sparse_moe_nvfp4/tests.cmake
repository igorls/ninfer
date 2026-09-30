# sparse_moe_nvfp4: Op qualification tests.
ninfer_add_op_test(ninfer_sparse_moe_nvfp4_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sparse_moe_nvfp4.cpp"
  LIBRARIES ninfer_ops)
