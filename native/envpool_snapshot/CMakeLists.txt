cmake_minimum_required(VERSION 3.24...4.4)
project(perpetua_envpool_snapshot LANGUAGES CXX)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "This optional binding is currently validated on Linux only")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Git REQUIRED)
include(FetchContent)

# Only headers/core are reused; do not build unrelated simulators or rendering.
FetchContent_Declare(envpool
  URL https://codeload.github.com/sail-sg/envpool/tar.gz/9c31c5478eb61d8f67f8c9a1ec2b37568f2c7ca3
  URL_HASH SHA256=0e297b30249c5d34d1b7ad0774d925d053c5d484b992270caacd9db2570917a8
  PATCH_COMMAND ${GIT_EXECUTABLE} apply ${CMAKE_CURRENT_SOURCE_DIR}/core.patch)
FetchContent_Declare(concurrentqueue
  URL https://codeload.github.com/cameron314/concurrentqueue/tar.gz/refs/tags/v1.0.5
  URL_HASH SHA256=4d6368a27492d86011fde5ca0cf386dce7c49cd425aa3d9b063ca6ec373a6ef3
  SOURCE_SUBDIR unused)
FetchContent_Declare(threadpool
  URL https://codeload.github.com/progschj/ThreadPool/tar.gz/9a42ec1329f259a5f4881a291db1dcb8f2ad9040
  URL_HASH SHA256=954e0ecdac1aa0da1e0fa78577ff0d352e53094df43762fbc1884f76a7e1dcd2)
FetchContent_Declare(absl
  URL https://codeload.github.com/abseil/abseil-cpp/tar.gz/refs/tags/20260817.0
  URL_HASH SHA256=f7e05179df39c45434cad433f5783840bb3788ef322976f9138bc6b72b3a107d)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(envpool concurrentqueue threadpool absl)

pybind11_add_module(perpetua_envpool_snapshot module.cc)
target_include_directories(perpetua_envpool_snapshot PRIVATE
  ${envpool_SOURCE_DIR} ${concurrentqueue_SOURCE_DIR} ${threadpool_SOURCE_DIR})
target_link_libraries(perpetua_envpool_snapshot PRIVATE absl::log absl::check)
install(TARGETS perpetua_envpool_snapshot DESTINATION .)
set(licenses share/perpetua-envpool-snapshot/licenses)
install(FILES ${envpool_SOURCE_DIR}/LICENSE DESTINATION ${licenses} RENAME envpool.txt)
install(FILES ${absl_SOURCE_DIR}/LICENSE DESTINATION ${licenses} RENAME absl.txt)
install(FILES ${concurrentqueue_SOURCE_DIR}/LICENSE.md
  DESTINATION ${licenses} RENAME concurrentqueue.txt)
# The semaphore's separate zlib license is embedded in its source header.
install(FILES ${concurrentqueue_SOURCE_DIR}/lightweightsemaphore.h DESTINATION ${licenses})
install(FILES ${threadpool_SOURCE_DIR}/COPYING DESTINATION ${licenses} RENAME threadpool.txt)
execute_process(COMMAND ${Python_EXECUTABLE} -c
  "import importlib.metadata as m; d=m.distribution('pybind11'); print(next(str(d.locate_file(f)) for f in d.files if str(f).endswith('/LICENSE')))"
  OUTPUT_VARIABLE pybind_license OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
install(FILES ${pybind_license} DESTINATION ${licenses} RENAME pybind11.txt)
