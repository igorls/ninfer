ninfer_add_test(ninfer_qwen4_exp_loading_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading_real.cpp"
  LIBRARIES ninfer_model_runtime ninfer_model_loading ninfer_core)

ninfer_add_test(ninfer_qwen4_exp_engine_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_real.cpp"
  LIBRARIES ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen4_exp_state_image_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image.cpp"
  LIBRARIES ninfer_model_runtime ninfer_core)

set_tests_properties(
  ninfer_qwen4_exp_loading_real_test
  ninfer_qwen4_exp_engine_real_test
  ninfer_qwen4_exp_state_image_test
  PROPERTIES SKIP_RETURN_CODE 77)
