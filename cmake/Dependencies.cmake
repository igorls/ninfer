find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Media decode needs FFmpeg and media acquisition needs libcurl. Linux resolves both through
# pkg-config. The Windows shared distributions (BtbN FFmpeg, a local libcurl install) ship MSVC
# import libraries whose .pc files describe MinGW link lines, so on Windows they are located
# from FFMPEG_ROOT and CURL_ROOT instead. Either way, consumers link ninfer::ffmpeg and
# ninfer::libcurl.
if(WIN32)
  # Release builds bundle FFmpeg, so the default is the LGPL distribution; never ship a GPL one.
  set(FFMPEG_ROOT
    "P:/third_party/ffmpeg-lgpl/ffmpeg-n9.0.2-3-ga5923073bf-win64-lgpl-shared-9.0"
    CACHE PATH "FFmpeg shared distribution root (include/, lib/, bin/)")
  find_path(FFMPEG_INCLUDE_DIR NAMES libavformat/avformat.h
    HINTS ${FFMPEG_ROOT}/include NO_DEFAULT_PATH)
  if(NOT FFMPEG_INCLUDE_DIR)
    message(FATAL_ERROR "FFmpeg headers not found under ${FFMPEG_ROOT}/include; set FFMPEG_ROOT")
  endif()
  set(ninfer_ffmpeg_libraries "")
  foreach(component IN ITEMS avformat avcodec avutil swscale)
    string(TOUPPER ${component} upper)
    find_library(${upper}_LIBRARY NAMES ${component}
      HINTS ${FFMPEG_ROOT}/lib NO_DEFAULT_PATH)
    if(NOT ${upper}_LIBRARY)
      message(FATAL_ERROR "FFmpeg ${component} import library not found under ${FFMPEG_ROOT}/lib")
    endif()
    list(APPEND ninfer_ffmpeg_libraries ${${upper}_LIBRARY})
  endforeach()
  add_library(ninfer::ffmpeg INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::ffmpeg INTERFACE ${FFMPEG_INCLUDE_DIR})
  target_link_libraries(ninfer::ffmpeg INTERFACE ${ninfer_ffmpeg_libraries})
  message(STATUS "FFmpeg: ${FFMPEG_ROOT}")
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET GLOBAL
    libavformat libavcodec libavutil libswscale)
  add_library(ninfer::ffmpeg ALIAS PkgConfig::FFMPEG)
endif()

# Repository-pinned header dependencies. No configure-time downloads.
add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

# Source base for the custom-template frontend; consumers will link it explicitly.
add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)

if(NINFER_BUILD_PRODUCT_SUPPORT)
  # Media acquisition uses CURLOPT_PROTOCOLS_STR and CURLOPT_REDIR_PROTOCOLS_STR,
  # introduced in libcurl 7.85 (not merely the version of the maintainer environment).
  if(WIN32)
    set(CURL_ROOT "P:/third_party/curl-inst" CACHE PATH "libcurl install root (include/, lib/, bin/)")
    find_path(CURL_INCLUDE_DIR NAMES curl/curl.h HINTS ${CURL_ROOT}/include NO_DEFAULT_PATH)
    find_library(CURL_LIBRARY NAMES libcurl_imp libcurl curl HINTS ${CURL_ROOT}/lib NO_DEFAULT_PATH)
    if(NOT CURL_INCLUDE_DIR OR NOT CURL_LIBRARY)
      message(FATAL_ERROR "libcurl not found under ${CURL_ROOT}; set CURL_ROOT")
    endif()
    file(STRINGS ${CURL_INCLUDE_DIR}/curl/curlver.h curl_version_line
      REGEX "^#define LIBCURL_VERSION \"[0-9.]+")
    string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" curl_version "${curl_version_line}")
    if(curl_version VERSION_LESS 7.85)
      message(FATAL_ERROR "libcurl >= 7.85 is required; ${CURL_ROOT} has ${curl_version}")
    endif()
    add_library(ninfer::libcurl INTERFACE IMPORTED GLOBAL)
    target_include_directories(ninfer::libcurl INTERFACE ${CURL_INCLUDE_DIR})
    target_link_libraries(ninfer::libcurl INTERFACE ${CURL_LIBRARY})
    message(STATUS "libcurl ${curl_version}: ${CURL_ROOT}")
  else()
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET GLOBAL libcurl>=7.85)
    add_library(ninfer::libcurl ALIAS PkgConfig::LIBCURL)
  endif()
  add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  add_subdirectory(third_party/spdlog)
endif()

# Windows has no rpath: CTest prepends the directories holding the shared libraries that
# executables load at run time to PATH.
set(NINFER_RUNTIME_DLL_DIRS "${CUDAToolkit_BIN_DIR}" "${CUDAToolkit_BIN_DIR}/x64")
if(WIN32)
  list(APPEND NINFER_RUNTIME_DLL_DIRS "${FFMPEG_ROOT}/bin")
  if(NINFER_BUILD_PRODUCT_SUPPORT)
    list(APPEND NINFER_RUNTIME_DLL_DIRS "${CURL_ROOT}/bin")
  endif()
endif()
