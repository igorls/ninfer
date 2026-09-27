ninfer_add_test(ninfer_admission_policy_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_admission_policy.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_context_cost_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_context_cost.cpp"
  LIBRARIES ninfer_runtime_support ninfer::json)

ninfer_add_test(ninfer_resource_manager_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_resource_manager.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_token_logprobs_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_token_logprobs.cpp")

ninfer_add_test(ninfer_structured_output_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_structured_output.cpp"
  LIBRARIES ninfer_runtime_support ninfer::json)

ninfer_add_test(ninfer_required_tool_call_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_required_tool_call.cpp"
  LIBRARIES ninfer_model_runtime ninfer_runtime_support ninfer::json)

ninfer_add_test(ninfer_kv_capacity_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_kv_capacity.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_sampling_defaults_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_sampling_defaults.cpp"
  LIBRARIES ninfer_engine ninfer_core)
