cmake_minimum_required(VERSION 3.20)

project(mixed_precision_fp
    VERSION 0.1
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Paths
set(HDNUM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/hdnum")
set(CPFLOAT_DIR "${HDNUM_DIR}/cpfloat")


# Main sandbox executable.
add_executable(hdnum_sandbox
    examples/hdnum_sandbox.cc
)

target_include_directories(hdnum_sandbox PRIVATE
    "${HDNUM_DIR}"
    "${CPFLOAT_DIR}/build/include"
)

target_compile_definitions(hdnum_sandbox PRIVATE
    HDNUM_HAS_GMP=1
    HDNUM_HAS_CPFLOAT=1
)

target_link_libraries(hdnum_sandbox PRIVATE
    "${CPFLOAT_DIR}/build/lib/libcpfloat.a"
    gmpxx
    gmp
)


# test conversions executable
add_executable(test_conversion
    tests/test_conversion.cc
)

target_include_directories(test_conversion PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${HDNUM_DIR}"
    "${CPFLOAT_DIR}/build/include"
)

target_compile_definitions(test_conversion PRIVATE
    HDNUM_HAS_GMP=1
    HDNUM_HAS_CPFLOAT=1
)

target_link_libraries(test_conversion PRIVATE
    "${CPFLOAT_DIR}/build/lib/libcpfloat.a"
    gmpxx
    gmp
)