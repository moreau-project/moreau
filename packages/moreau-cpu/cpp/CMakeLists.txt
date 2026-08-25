cmake_minimum_required(VERSION 3.14)
project(moreau_active_set LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# =============================================================================
# Static library: libmoreau_active_set.a
# =============================================================================

add_library(moreau_active_set STATIC
    src/solver/active_set/factorization.cpp
    src/solver/active_set/core.cpp
    src/solver/active_set/transform.cpp
    src/solver/active_set/solver.cpp
    src/solver/active_set/backward.cpp
    src/c_api_active_set.cpp
)

target_include_directories(moreau_active_set PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(Threads REQUIRED)
target_link_libraries(moreau_active_set PUBLIC Threads::Threads)

# Install the static library for cargo to link
install(TARGETS moreau_active_set
    ARCHIVE DESTINATION lib
)

# =============================================================================
# Tests (optional)
# =============================================================================

option(MOREAU_BUILD_TESTS "Build tests" ON)

if(MOREAU_BUILD_TESTS)
    find_package(GTest QUIET)
    if(GTest_FOUND)
        enable_testing()

        add_executable(test_active_set tests/test_active_set.cpp)
        target_link_libraries(test_active_set PRIVATE moreau_active_set GTest::gtest GTest::gtest_main)
        add_test(NAME test_active_set COMMAND test_active_set)

        message(STATUS "Tests enabled")
    else()
        message(STATUS "GTest not found - tests will not be built")
    endif()
endif()
