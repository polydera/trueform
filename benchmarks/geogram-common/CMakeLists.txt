# Geogram benchmark common library
# Provides shared Geogram utilities and conversions

add_library(benchmark_geogram_common INTERFACE)
target_include_directories(benchmark_geogram_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(benchmark_geogram_common INTERFACE benchmark_common geogram)
