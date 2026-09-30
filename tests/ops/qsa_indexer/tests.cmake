# qsa_indexer: Op qualification tests.
ninfer_add_op_test(ninfer_qsa_indexer_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_qsa_indexer.cpp"
  LIBRARIES ninfer_ops)
