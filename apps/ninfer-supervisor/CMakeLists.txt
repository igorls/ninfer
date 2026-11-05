if(NOT WIN32)
  return()
endif()

enable_language(RC)
set(NINFER_SUPERVISOR_DASHBOARD_HTML "${CMAKE_CURRENT_SOURCE_DIR}/dashboard.html")
# Visual Studio does not emit OBJECT_DEPENDS on ResourceCompile items. Reconfigure
# on payload edits and change the RC input too, so every generator rebuilds it.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${NINFER_SUPERVISOR_DASHBOARD_HTML}")
file(SHA256 "${NINFER_SUPERVISOR_DASHBOARD_HTML}" NINFER_SUPERVISOR_DASHBOARD_SHA256)
configure_file(dashboard.rc.in "${CMAKE_CURRENT_BINARY_DIR}/dashboard.rc"
  @ONLY NEWLINE_STYLE UNIX)
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/dashboard.rc" PROPERTIES
  OBJECT_DEPENDS "${NINFER_SUPERVISOR_DASHBOARD_HTML}")

# WIN32 links this for the Windows subsystem so no console window appears when
# the supervisor autostarts at login. It still has a main() rather than a
# WinMain(), so the entry point is set back explicitly; main() adopts the
# parent terminal when the binary is run by hand (see attach_parent_console_if_any).
add_executable(ninfer-supervisor WIN32
  main.cpp
  dashboard.cpp
  "${CMAKE_CURRENT_BINARY_DIR}/dashboard.rc"
  engine_child.cpp
  collector.cpp
  server.cpp
  frontend_server.cpp
  engine_proxy.cpp
  tray.cpp)
target_include_directories(ninfer-supervisor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ninfer-supervisor PRIVATE ninfer::httplib ninfer::json ninfer_product_config CUDA::cudart)
target_link_options(ninfer-supervisor PRIVATE
  "LINKER:/ENTRY:mainCRTStartup"
  "LINKER:/DEFAULTLIB:dxgi"
  "LINKER:/DEFAULTLIB:pdh"
  "LINKER:/DEFAULTLIB:ole32"
  "LINKER:/DEFAULTLIB:oleaut32"
  "LINKER:/DEFAULTLIB:uuid"
  "LINKER:/DEFAULTLIB:shell32"
  "LINKER:/DEFAULTLIB:user32"
  "LINKER:/DEFAULTLIB:gdi32"
  "LINKER:/DEFAULTLIB:advapi32"
  "LINKER:/DEFAULTLIB:ws2_32"
  "LINKER:/DEFAULTLIB:crypt32")
# One setting for every translation unit that includes the header-only httplib.
target_compile_definitions(ninfer-supervisor PRIVATE
  CPPHTTPLIB_NO_EXCEPTIONS
  WIN32_LEAN_AND_MEAN
  NOMINMAX
  _WIN32_WINNT=0x0A00)
