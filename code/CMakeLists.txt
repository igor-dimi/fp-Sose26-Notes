cmake_minimum_required(VERSION 3.20)

project(mixed_precision_fp
    VERSION 0.1
    LANGUAGES CXX
)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# --------------------------------------------------------------------
# External dependency paths
# --------------------------------------------------------------------

set(HDNUM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/hdnum")
set(CPFLOAT_DIR "${HDNUM_DIR}/cpfloat")

set(CPFLOAT_INCLUDE_DIR "${CPFLOAT_DIR}/build/include")
set(CPFLOAT_LIBRARY "${CPFLOAT_DIR}/build/lib/libcpfloat.a")

# --------------------------------------------------------------------
# Shared project target
# --------------------------------------------------------------------
# This target collects all include paths, compile definitions, and
# libraries needed by your examples, tests, and experiments.

add_library(mixed_precision_core INTERFACE)

target_include_directories(mixed_precision_core
    INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${HDNUM_DIR}"
        "${CPFLOAT_INCLUDE_DIR}"
)

target_compile_definitions(mixed_precision_core
    INTERFACE
        HDNUM_HAS_GMP=1
        HDNUM_HAS_CPFLOAT=1
)

target_link_libraries(mixed_precision_core
    INTERFACE
        "${CPFLOAT_LIBRARY}"
        gmpxx
        gmp
)

target_compile_features(mixed_precision_core
    INTERFACE
        cxx_std_17
)

# --------------------------------------------------------------------
# Compiler warnings
# --------------------------------------------------------------------

add_library(mixed_precision_warnings INTERFACE)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mixed_precision_warnings
        INTERFACE
            -Wall
            -Wextra
            -Wpedantic
    )
elseif(MSVC)
    target_compile_options(mixed_precision_warnings
        INTERFACE
            /W4
    )
endif()

# --------------------------------------------------------------------
# Helper: examples and demos
# --------------------------------------------------------------------
# In the current directory, every file matching
#
#   ex_*.cc, ex_*.cpp, demo_*.cc, demo_*.cpp
#
# becomes an executable.

function(add_mixed_precision_examples target_prefix)
    set(options)
    set(one_value_args)
    set(multi_value_args LINK_LIBRARIES INCLUDE_DIRS SOURCES)

    cmake_parse_arguments(ARG
        "${options}"
        "${one_value_args}"
        "${multi_value_args}"
        ${ARGN}
    )

    file(GLOB example_sources CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/ex_*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/ex_*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/demo_*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/demo_*.cpp"
    )

    foreach(source_file IN LISTS example_sources)
        get_filename_component(source_name "${source_file}" NAME_WE)
        set(target_name "${target_prefix}_${source_name}")

        add_executable(${target_name}
            "${source_file}"
            ${ARG_SOURCES}
        )

        target_link_libraries(${target_name}
            PRIVATE
                mixed_precision_core
                mixed_precision_warnings
                ${ARG_LINK_LIBRARIES}
        )

        if(ARG_INCLUDE_DIRS)
            target_include_directories(${target_name}
                PRIVATE
                    ${ARG_INCLUDE_DIRS}
            )
        endif()
    endforeach()
endfunction()

# --------------------------------------------------------------------
# Helper: tests
# --------------------------------------------------------------------
# In the current directory, every file matching
#
#   test_*.cc, test_*.cpp
#
# becomes an executable and is also registered as a CTest test.

function(add_mixed_precision_tests target_prefix)
    set(options)
    set(one_value_args)
    set(multi_value_args LINK_LIBRARIES INCLUDE_DIRS SOURCES)

    cmake_parse_arguments(ARG
        "${options}"
        "${one_value_args}"
        "${multi_value_args}"
        ${ARGN}
    )

    file(GLOB test_sources CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp"
    )

    foreach(source_file IN LISTS test_sources)
        get_filename_component(source_name "${source_file}" NAME_WE)
        set(target_name "${target_prefix}_${source_name}")

        add_executable(${target_name}
            "${source_file}"
            ${ARG_SOURCES}
        )

        target_link_libraries(${target_name}
            PRIVATE
                mixed_precision_core
                mixed_precision_warnings
                ${ARG_LINK_LIBRARIES}
        )

        if(ARG_INCLUDE_DIRS)
            target_include_directories(${target_name}
                PRIVATE
                    ${ARG_INCLUDE_DIRS}
            )
        endif()

        add_test(
            NAME ${target_name}
            COMMAND ${target_name}
        )
    endforeach()
endfunction()

# --------------------------------------------------------------------
# Helper: experiments
# --------------------------------------------------------------------
# In the current directory, every file matching
#
#   exp_*.cc, exp_*.cpp
#
# becomes an executable.

function(add_mixed_precision_experiments target_prefix)
    set(options)
    set(one_value_args)
    set(multi_value_args LINK_LIBRARIES INCLUDE_DIRS SOURCES)

    cmake_parse_arguments(ARG
        "${options}"
        "${one_value_args}"
        "${multi_value_args}"
        ${ARGN}
    )

    file(GLOB experiment_sources CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/exp_*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/exp_*.cpp"
    )

    foreach(source_file IN LISTS experiment_sources)
        get_filename_component(source_name "${source_file}" NAME_WE)
        set(target_name "${target_prefix}_${source_name}")

        add_executable(${target_name}
            "${source_file}"
            ${ARG_SOURCES}
        )

        target_link_libraries(${target_name}
            PRIVATE
                mixed_precision_core
                mixed_precision_warnings
                ${ARG_LINK_LIBRARIES}
        )

        if(ARG_INCLUDE_DIRS)
            target_include_directories(${target_name}
                PRIVATE
                    ${ARG_INCLUDE_DIRS}
            )
        endif()
    endforeach()
endfunction()

# --------------------------------------------------------------------
# Subdirectories
# --------------------------------------------------------------------

add_subdirectory(examples)
add_subdirectory(tests)
add_subdirectory(experiments)