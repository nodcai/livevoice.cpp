cmake_minimum_required(VERSION 3.18)
project(livevoice
        VERSION   0.1.0
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LIVEVOICE_BUILD_TESTS    "Build tests"          OFF)
option(LIVEVOICE_BUILD_EXAMPLES "Build CLI tools"      ON)
option(LIVEVOICE_SHARED         "Build shared library" OFF)

# ---------- ggml ----------
# A parent project that already has a ggml target (e.g. an app embedding
# several ggml-based libraries) keeps its own; otherwise use the pinned
# submodule. Backends follow ggml's own GGML_* options (Metal is on by
# default on Apple).
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(NOT TARGET ggml)
    add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)
endif()

# ---------- liblivevoice ----------
set(LIVEVOICE_SOURCES
    src/common.cpp
    src/backend.cpp
    src/gguf_loader.cpp
    src/tokenizer.cpp
)

if(LIVEVOICE_SHARED)
    add_library(livevoice SHARED ${LIVEVOICE_SOURCES})
    target_compile_definitions(livevoice PUBLIC LIVEVOICE_SHARED PRIVATE LIVEVOICE_BUILD)
else()
    add_library(livevoice STATIC ${LIVEVOICE_SOURCES})
endif()

target_include_directories(livevoice
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
target_link_libraries(livevoice PUBLIC ggml)
target_compile_options(livevoice PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -Wno-unused-parameter>
)

# ---------- subdirs ----------
if(LIVEVOICE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(LIVEVOICE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------- install ----------
include(GNUInstallDirs)
install(TARGETS livevoice
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES include/livevoice.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
