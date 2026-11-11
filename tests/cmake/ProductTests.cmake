ninfer_add_test(ninfer_media_decode_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_media_decode.cpp"
  LIBRARIES ninfer_media_decode)

ninfer_add_test(ninfer_prompt_input_test
  NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_prompt_input.cpp"
  LIBRARIES ninfer_product_prompt_input)

ninfer_add_test(ninfer_pretty_logging_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_pretty_logging.cpp"
  LIBRARIES ninfer_product_logging)

ninfer_add_test(ninfer_perplexity_evaluation_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_perplexity_evaluation.cpp"
          ${PROJECT_SOURCE_DIR}/apps/perplexity/evaluation.cpp
          ${PROJECT_SOURCE_DIR}/apps/perplexity/options.cpp
          ${PROJECT_SOURCE_DIR}/apps/perplexity/report.cpp
  LIBRARIES ninfer_core ninfer_product_config ninfer::json)

target_include_directories(ninfer_perplexity_evaluation_test PRIVATE
  ${PROJECT_SOURCE_DIR}/apps/perplexity)

ninfer_add_test(ninfer_cli_options_test
  NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_cli_options.cpp"
  LIBRARIES ninfer_product_config)

ninfer_add_test(ninfer_fork_startup_options_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_fork_startup_options.cpp"
  LIBRARIES ninfer_product_config)

ninfer_add_test(ninfer_openai_schema_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_openai_schema.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_openai_responses_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_openai_responses.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_openai_responses_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_openai_responses_store.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_anthropic_schema_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_anthropic_schema.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_typesafe_schema_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_typesafe_schema.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_rerank_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_ninfer_rerank.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_serve_options_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_serve_options.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_request_log_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_request_log.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_expert_route_recorder_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_expert_route_recorder.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_device_snapshot_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_device_snapshot_cache.cpp"
  LIBRARIES ninfer_core)
set_tests_properties(ninfer_device_snapshot_cache_test PROPERTIES TIMEOUT 10)
ninfer_add_test(ninfer_metrics_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_metrics.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_http_error_handler_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_error_handler.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_http_transport_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_transport.cpp"
  LIBRARIES ninfer_serve)

if(WIN32)
  find_program(NINFER_TEST_NODE_EXECUTABLE NAMES node)
  if(NINFER_TEST_NODE_EXECUTABLE)
    add_test(NAME ninfer_supervisor_dashboard_test
      COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/../test_supervisor_dashboard.py"
        --node "${NINFER_TEST_NODE_EXECUTABLE}"
        --html "${PROJECT_SOURCE_DIR}/apps/ninfer-supervisor/dashboard.html")
  endif()
  ninfer_add_test(ninfer_supervisor_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_ninfer_supervisor.cpp"
            "${PROJECT_SOURCE_DIR}/apps/ninfer-supervisor/frontend_server.cpp"
            "${PROJECT_SOURCE_DIR}/apps/ninfer-supervisor/engine_proxy.cpp"
    LIBRARIES ninfer::httplib ninfer::json ninfer_product_config)
  target_include_directories(ninfer_supervisor_test PRIVATE
    ${PROJECT_SOURCE_DIR}/apps/ninfer-supervisor)
  # Same httplib configuration as the supervisor, whose frontend server this test links.
  target_compile_definitions(ninfer_supervisor_test PRIVATE
    WIN32_LEAN_AND_MEAN NOMINMAX)
endif()
if(TARGET ninfer-serve)
  add_test(NAME ninfer_serve_shutdown_real_test
    COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/../test_serve_shutdown_real.py"
      $<TARGET_FILE:ninfer-serve>)
  set_tests_properties(ninfer_serve_shutdown_real_test PROPERTIES
    SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 150)
endif()

ninfer_add_test(ninfer_tool_streaming_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_tool_streaming.cpp"
  LIBRARIES ninfer_serve ninfer_model_execution ninfer::json)
