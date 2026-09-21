/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */
#include "benchmark_black_box.hpp"

#include "trueform/cpp/core/elementwise.hpp"
#include "trueform/cpp/core/histogram.hpp"
#include "trueform/cpp/core/nd_array_indexing.hpp"
#include "trueform/cpp/core/nd_array_sorting.hpp"
#include "trueform/cpp/core/nd_array_structure.hpp"
#include "trueform/cpp/core/reductions.hpp"

#include <tbb/global_control.h>
#include <tbb/info.h>
#include <tbb/task_arena.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;
constexpr double minimum_sample_milliseconds = 10.0;
constexpr std::size_t maximum_batch_operations = 1U << 20U;

struct options {
  std::string filter;
  int samples = 5;
  int thread_cap = 2;
  bool list = false;
};

struct result_summary {
  std::string shape;
  std::size_t length = 0;
  std::uint64_t digest = 0;
};

struct benchmark_case {
  std::function<result_summary()> validation_run;
  std::function<void(std::size_t)> timed_run;
  std::string expected_shape;
  std::size_t expected_length = 0;
};

struct case_spec {
  std::string name;
  std::string dtype;
  std::string parameters;
  std::function<benchmark_case()> setup;
};

auto hash_bytes(std::uint64_t hash, const void *data, std::size_t size)
    -> std::uint64_t {
  const auto *bytes = static_cast<const unsigned char *>(data);
  for (std::size_t index = 0; index < size; ++index) {
    hash ^= bytes[index];
    hash *= fnv_prime;
  }
  return hash;
}

template <typename T>
auto hash_scalar(std::uint64_t hash, const T &value) -> std::uint64_t {
  return hash_bytes(hash, &value, sizeof(T));
}

auto shape_string(const tf::small_vector<int, 3> &shape) -> std::string {
  std::ostringstream output;
  output << '[';
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index != 0)
      output << ',';
    output << shape[index];
  }
  output << ']';
  return output.str();
}

template <typename T>
auto summarize(const tf::cpp::nd_array<T> &array) -> result_summary {
  auto digest = fnv_offset;
  for (const auto dimension : array.raw_shape())
    digest = hash_scalar(digest, dimension);
  digest = hash_scalar(digest, array.length());
  if (array.length() != 0)
    digest = hash_bytes(digest, array.raw_data(), array.length() * sizeof(T));
  return {shape_string(array.raw_shape()), array.length(), digest};
}

auto summarize(const tf::cpp::histogram_result<std::int32_t> &result)
    -> result_summary {
  const auto counts = summarize(result.counts);
  const auto edges = summarize(result.edges);
  auto digest = hash_scalar(fnv_offset, counts.digest);
  digest = hash_scalar(digest, edges.digest);
  return {"counts=" + counts.shape + ";edges=" + edges.shape,
          counts.length + edges.length, digest};
}

template <typename T>
auto escape_storage(const tf::cpp::nd_array<T> &array) -> void {
  tf_cpp_benchmark_black_box(array.raw_data(), array.length() * sizeof(T));
}

auto escape_storage(const tf::cpp::histogram_result<std::int32_t> &result)
    -> void {
  escape_storage(result.counts);
  escape_storage(result.edges);
}

template <typename Invoke>
auto make_benchmark_case(Invoke invoke, std::string expected_shape,
                         std::size_t expected_length) -> benchmark_case {
  auto validation_run = [invoke]() mutable {
    const auto result = invoke();
    return summarize(result);
  };
  auto timed_run = [invoke](std::size_t operations) mutable {
    for (std::size_t operation = 0; operation < operations; ++operation) {
      const auto result = invoke();
      escape_storage(result);
      // Each complete result escapes while alive, then is destroyed before the
      // next operation. Allocation and destruction are both timed.
    }
  };
  return {std::move(validation_run), std::move(timed_run),
          std::move(expected_shape), expected_length};
}

auto shape_length(const tf::small_vector<int, 3> &shape) -> std::size_t {
  auto length = std::size_t{1};
  for (const auto dimension : shape)
    length *= static_cast<std::size_t>(dimension);
  return length;
}

template <typename T, typename Generator>
auto make_array(tf::small_vector<int, 3> shape, Generator generator)
    -> tf::cpp::nd_array<T> {
  const auto length = shape_length(shape);
  tf::buffer<T> buffer;
  buffer.allocate(length);
  auto *data = buffer.data();
  for (std::size_t index = 0; index < length; ++index)
    data[index] = generator(index);
  return tf::cpp::nd_array<T>::from_buffer(std::move(buffer), std::move(shape));
}

auto float_value(std::size_t index) -> float {
  const auto value = static_cast<int>((index * 48271ULL + 17ULL) % 2003ULL);
  return static_cast<float>(value - 1001) / 1001.0F;
}

template <typename T> auto deep_copy_value(std::size_t index) -> T {
  if constexpr (std::is_same_v<T, std::int8_t>)
    return static_cast<std::int8_t>(static_cast<int>(index % 127U) - 63);
  return static_cast<T>(float_value(index));
}

auto histogram_value(std::size_t index) -> float {
  const auto value = (index * 48271ULL + 17ULL) % 1000003ULL;
  return static_cast<float>(value) / 1000002.0F;
}

auto index_selection(int count, int modulus, int offset)
    -> std::vector<std::int32_t> {
  std::vector<std::int32_t> indices(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index)
    indices[static_cast<std::size_t>(index)] =
        static_cast<std::int32_t>((index * 37 + offset) % modulus);
  return indices;
}

auto selected_rows(int rows, int period, int selected_per_period)
    -> std::size_t {
  auto count = std::size_t{0};
  for (int row = 0; row < rows; ++row)
    if (row % period < selected_per_period)
      ++count;
  return count;
}

auto make_cases() -> std::vector<case_spec> {
  std::vector<case_spec> cases;

  cases.push_back({"add_equal", "float32", "a=[1000000,3];b=[1000000,3]", [] {
                     constexpr int rows = 1000000;
                     auto a = make_array<float>({rows, 3}, float_value);
                     auto b =
                         make_array<float>({rows, 3}, [](std::size_t index) {
                           return float_value(index + 101);
                         });
                     return make_benchmark_case(
                         [a = std::move(a), b = std::move(b)] {
                           return tf::cpp::add(a, b);
                         },
                         "[1000000,3]", static_cast<std::size_t>(rows) * 3);
                   }});

  cases.push_back({"add_broadcast_vec3", "float32", "a=[1000000,3];b=[3]", [] {
                     constexpr int rows = 1000000;
                     auto a = make_array<float>({rows, 3}, float_value);
                     auto b = make_array<float>({3}, [](std::size_t index) {
                       return static_cast<float>(index + 1) * 0.25F;
                     });
                     return make_benchmark_case(
                         [a = std::move(a), b = std::move(b)] {
                           return tf::cpp::add(a, b);
                         },
                         "[1000000,3]", static_cast<std::size_t>(rows) * 3);
                   }});

  cases.push_back(
      {"axis_sum_small_output", "float32", "shape=[64,65536];axis=1;output=64",
       [] {
         constexpr int rows = 64;
         constexpr int inner = 65536;
         auto input = make_array<float>({rows, inner}, float_value);
         return make_benchmark_case(
             [input = std::move(input)] { return tf::cpp::sum(input, 1); },
             "[64]", rows);
       }});

  cases.push_back({"dot_large_inner_small_output", "float32",
                   "a=[64,65536];b=[64,65536];output=64", [] {
                     constexpr int rows = 64;
                     constexpr int inner = 65536;
                     auto a = make_array<float>({rows, inner}, float_value);
                     auto b = make_array<float>(
                         {rows, inner}, [](std::size_t index) {
                           return float_value(index + 313);
                         });
                     return make_benchmark_case(
                         [a = std::move(a), b = std::move(b)] {
                           return tf::cpp::dot(a, b);
                         },
                         "[64]", rows);
                   }});

  cases.push_back({"mat_mul_large_shared_small_output", "float32",
                   "a=[32,20000];b=[20000,32];output=[32,32]", [] {
                     constexpr int rows = 32;
                     constexpr int shared = 20000;
                     constexpr int columns = 32;
                     auto a = make_array<float>({rows, shared}, float_value);
                     auto b = make_array<float>(
                         {shared, columns}, [](std::size_t index) {
                           return float_value(index + 719);
                         });
                     return make_benchmark_case(
                         [a = std::move(a), b = std::move(b)] {
                           return tf::cpp::mat_mul(a, b);
                         },
                         "[32,32]", static_cast<std::size_t>(rows) * columns);
                   }});

  cases.push_back({"normalize_axis_small_carrier", "float32",
                   "shape=[64,65536];axis=1", [] {
                     constexpr int rows = 64;
                     constexpr int inner = 65536;
                     auto input = make_array<float>(
                         {rows, inner}, [](std::size_t index) {
                           return float_value(index) + 1.25F;
                         });
                     return make_benchmark_case(
                         [input = std::move(input)] {
                           return tf::cpp::normalize(input, 1);
                         },
                         "[64,65536]", static_cast<std::size_t>(rows) * inner);
                   }});

  const auto add_equal_width_histogram = [&cases](const char *name, int bins) {
    cases.push_back(
        {name, "float32",
         "values=2000000;bins=" + std::to_string(bins) + ";range=[0,1]",
         [bins] {
           constexpr int value_count = 2000000;
           auto values = make_array<float>({value_count}, histogram_value);
           return make_benchmark_case(
               [values = std::move(values), bins] {
                 return tf::cpp::histogram_equal_width(values, bins, 0.0F,
                                                       1.0F);
               },
               "counts=[" + std::to_string(bins) + "];edges=[" +
                   std::to_string(bins + 1) + "]",
               static_cast<std::size_t>(bins) * 2 + 1);
         }});
  };
  add_equal_width_histogram("histogram_equal_width_256", 256);
  add_equal_width_histogram("histogram_equal_width_8192", 8192);

  const auto add_explicit_edge_histogram = [&cases](const char *name,
                                                    int bins) {
    cases.push_back(
        {name, "float32",
         "values=2000000;explicit_edges=" + std::to_string(bins + 1), [bins] {
           constexpr int value_count = 2000000;
           auto values = make_array<float>({value_count}, histogram_value);
           auto edges =
               make_array<float>({bins + 1}, [bins](std::size_t index) {
                 const auto ratio =
                     static_cast<float>(index) / static_cast<float>(bins);
                 return ratio * ratio;
               });
           return make_benchmark_case(
               [values = std::move(values), edges = std::move(edges)] {
                 return tf::cpp::histogram_edges(values, edges);
               },
               "counts=[" + std::to_string(bins) + "];edges=[" +
                   std::to_string(bins + 1) + "]",
               static_cast<std::size_t>(bins) * 2 + 1);
         }});
  };
  add_explicit_edge_histogram("histogram_explicit_edges_256", 256);
  add_explicit_edge_histogram("histogram_explicit_edges_8192", 8192);

  cases.push_back(
      {"stack_trailing_axis", "float32", "arrays=8;input=[250000,3];axis=2",
       [] {
         constexpr int rows = 250000;
         std::vector<tf::cpp::nd_array<float>> arrays;
         arrays.reserve(8);
         for (int array_index = 0; array_index < 8; ++array_index) {
           arrays.push_back(
               make_array<float>({rows, 3}, [array_index](std::size_t index) {
                 return float_value(index +
                                    static_cast<std::size_t>(array_index) * 97);
               }));
         }
         return make_benchmark_case(
             [arrays = std::move(arrays)] { return tf::cpp::stack(arrays, 2); },
             "[250000,3,8]", static_cast<std::size_t>(rows) * 3 * 8);
       }});

  cases.push_back(
      {"concatenate_trailing_axis", "float32",
       "arrays=8;input=[250000,1];axis=1", [] {
         constexpr int rows = 250000;
         std::vector<tf::cpp::nd_array<float>> arrays;
         arrays.reserve(8);
         for (int array_index = 0; array_index < 8; ++array_index) {
           arrays.push_back(
               make_array<float>({rows, 1}, [array_index](std::size_t index) {
                 return float_value(
                     index + static_cast<std::size_t>(array_index) * 193);
               }));
         }
         return make_benchmark_case(
             [arrays = std::move(arrays)] {
               return tf::cpp::concatenate(arrays, 1);
             },
             "[250000,8]", static_cast<std::size_t>(rows) * 8);
       }});

  cases.push_back(
      {"transpose_3d", "float32", "input=[256,256,32];axes=[2,0,1]", [] {
         auto input = make_array<float>({256, 256, 32}, float_value);
         return make_benchmark_case(
             [input = std::move(input)] {
               return tf::cpp::transpose(input, {2, 0, 1});
             },
             "[32,256,256]", 256ULL * 256ULL * 32ULL);
       }});

  cases.push_back(
      {"where_equal_shape", "float32", "condition/x/y=[4000000]", [] {
         constexpr int length = 4000000;
         auto condition =
             make_array<std::int8_t>({length}, [](std::size_t index) {
               return static_cast<std::int8_t>((index % 5) != 0);
             });
         auto x = make_array<float>({length}, float_value);
         auto y = make_array<float>({length}, [](std::size_t index) {
           return float_value(index + 401);
         });
         return make_benchmark_case(
             [condition = std::move(condition), x = std::move(x),
              y = std::move(y)] { return tf::cpp::where(condition, x, y); },
             "[4000000]", length);
       }});

  cases.push_back(
      {"sort_multidimensional_rows", "int32",
       "shape=[150000,8];lexicographic_rows;first_column_distinct", [] {
         constexpr int rows = 150000;
         constexpr int width = 8;
         auto input =
             make_array<std::int32_t>({rows, width}, [](std::size_t index) {
               const auto row = index / width;
               const auto column = index % width;
               return static_cast<std::int32_t>(
                   (row * 104729ULL + column * 8191ULL + 23ULL) % 1000003ULL);
             });
         return make_benchmark_case(
             [input = std::move(input)] { return tf::cpp::sort(input); },
             "[150000,8]", static_cast<std::size_t>(rows) * width);
       }});

  cases.push_back(
      {"sort_multidimensional_rows_common_prefix", "int32",
       "shape=[150000,8];lexicographic_rows;common_prefix=7", [] {
         constexpr int rows = 150000;
         constexpr int width = 8;
         auto input =
             make_array<std::int32_t>({rows, width}, [](std::size_t index) {
               const auto row = index / width;
               const auto column = index % width;
               if (column != width - 1)
                 return std::int32_t{17};
               return static_cast<std::int32_t>((row * 104729ULL + 23ULL) %
                                                1000003ULL);
             });
         return make_benchmark_case(
             [input = std::move(input)] { return tf::cpp::sort(input); },
             "[150000,8]", static_cast<std::size_t>(rows) * width);
       }});

  cases.push_back(
      {"multi_take_3d", "float32", "input=[256,256,32];selection=[192,192,24]",
       [] {
         auto input = make_array<float>({256, 256, 32}, float_value);
         std::vector<tf::cpp::multi_take_index> indices;
         indices.emplace_back(index_selection(192, 256, 11));
         indices.emplace_back(index_selection(192, 256, 29));
         indices.emplace_back(index_selection(24, 32, 7));
         return make_benchmark_case(
             [input = std::move(input), indices = std::move(indices)] {
               return tf::cpp::multi_take(input, indices);
             },
             "[192,192,24]", 192ULL * 192ULL * 24ULL);
       }});

  const auto add_boolean_index = [&cases](const char *name, int period,
                                          int selected_per_period) {
    constexpr int rows = 1000000;
    constexpr int width = 3;
    const auto selected = selected_rows(rows, period, selected_per_period);
    cases.push_back(
        {name, "float32",
         "input=[1000000,3];selected_rows=" + std::to_string(selected),
         [period, selected_per_period, selected] {
           auto input = make_array<float>({rows, width}, float_value);
           auto mask = make_array<std::int8_t>(
               {rows}, [period, selected_per_period](std::size_t index) {
                 return static_cast<std::int8_t>(
                     static_cast<int>(index %
                                      static_cast<std::size_t>(period)) <
                     selected_per_period);
               });
           return make_benchmark_case(
               [input = std::move(input), mask = std::move(mask)] {
                 return tf::cpp::boolean_index(input, mask);
               },
               "[" + std::to_string(selected) + ",3]", selected * width);
         }});
  };
  add_boolean_index("boolean_index_sparse", 100, 1);
  add_boolean_index("boolean_index_dense", 10, 9);

  const auto add_deep_copy = [&cases](auto value_type, const char *name,
                                      const char *dtype, int length) {
    using T = decltype(value_type);
    cases.push_back(
        {name, dtype,
         "shape=[" + std::to_string(length) + "];bytes=" +
             std::to_string(static_cast<std::size_t>(length) * sizeof(T)) +
             ";byte_policy=effective_parallelism<=1:serial;"
             "effective_parallelism>1:serial_through_1048576_bytes_then_"
             "parallel",
         [length] {
           auto input = make_array<T>({length}, deep_copy_value<T>);
           return make_benchmark_case(
               [input = std::move(input)] { return input.deep_copy(); },
               "[" + std::to_string(length) + "]",
               static_cast<std::size_t>(length));
         }});
  };
  add_deep_copy(float{}, "deep_copy_below_threshold", "float32", 999);
  add_deep_copy(float{}, "deep_copy_at_threshold", "float32", 1000);

  constexpr int serial_copy_max_bytes = 1 * 1024 * 1024;
  const auto add_deep_copy_crossover = [&](auto value_type, const char *dtype) {
    using T = decltype(value_type);
    const auto prefix = std::string("deep_copy_") + dtype;
    const auto serial_max = serial_copy_max_bytes / static_cast<int>(sizeof(T));
    add_deep_copy(T{}, (prefix + "_serial_max").c_str(), dtype, serial_max);
    add_deep_copy(T{}, (prefix + "_parallel_start").c_str(), dtype,
                  serial_max + 1);
  };
  add_deep_copy_crossover(std::int8_t{}, "int8");
  add_deep_copy_crossover(std::int32_t{}, "int32");
  add_deep_copy_crossover(float{}, "float32");
  add_deep_copy_crossover(double{}, "float64");

  const auto add_deep_copy_measured_sizes = [&](auto value_type,
                                                const char *dtype) {
    using T = decltype(value_type);
    for (const auto mebibytes : {2, 4, 8}) {
      const auto bytes = mebibytes * 1024 * 1024;
      const auto name = std::string("deep_copy_") + dtype + "_" +
                        std::to_string(mebibytes) + "mib";
      add_deep_copy(T{}, name.c_str(), dtype,
                    bytes / static_cast<int>(sizeof(T)));
    }
  };
  add_deep_copy_measured_sizes(std::int8_t{}, "int8");
  add_deep_copy_measured_sizes(std::int32_t{}, "int32");
  add_deep_copy_measured_sizes(float{}, "float32");
  add_deep_copy_measured_sizes(double{}, "float64");

  return cases;
}

auto parse_positive_int(const std::string &value, const char *name) -> int {
  if (value.empty())
    throw std::invalid_argument(std::string(name) + " requires a value");
  char *end = nullptr;
  const auto parsed = std::strtol(value.c_str(), &end, 10);
  if (*end != '\0' || parsed <= 0 ||
      parsed > static_cast<long>(std::numeric_limits<int>::max()))
    throw std::invalid_argument(std::string(name) +
                                " must be a positive integer");
  return static_cast<int>(parsed);
}

auto next_argument(int &index, int argc, char **argv, const char *name)
    -> std::string {
  if (index + 1 >= argc)
    throw std::invalid_argument(std::string(name) + " requires a value");
  return argv[++index];
}

auto parse_options(int argc, char **argv) -> options {
  options parsed;
  const auto hardware_threads = std::thread::hardware_concurrency();
  parsed.thread_cap = static_cast<int>(
      std::max(2U, hardware_threads == 0 ? 2U : hardware_threads));

  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: trueform_cpp_ndarray_benchmarks "
                   "[--filter TEXT] [--samples N] [--thread-cap N] [--list]\n";
      std::exit(0);
    }
    if (argument == "--list") {
      parsed.list = true;
    } else if (argument == "--filter") {
      parsed.filter = next_argument(index, argc, argv, "--filter");
    } else if (argument.rfind("--filter=", 0) == 0) {
      parsed.filter = argument.substr(9);
    } else if (argument == "--samples") {
      parsed.samples = parse_positive_int(
          next_argument(index, argc, argv, "--samples"), "--samples");
    } else if (argument.rfind("--samples=", 0) == 0) {
      parsed.samples = parse_positive_int(argument.substr(10), "--samples");
    } else if (argument == "--thread-cap" || argument == "--max-threads" ||
               argument == "--threads") {
      parsed.thread_cap = parse_positive_int(
          next_argument(index, argc, argv, argument.c_str()), argument.c_str());
    } else if (argument.rfind("--thread-cap=", 0) == 0) {
      parsed.thread_cap =
          parse_positive_int(argument.substr(13), "--thread-cap");
    } else if (argument.rfind("--max-threads=", 0) == 0) {
      parsed.thread_cap =
          parse_positive_int(argument.substr(14), "--max-threads");
    } else if (argument.rfind("--threads=", 0) == 0) {
      parsed.thread_cap = parse_positive_int(argument.substr(10), "--threads");
    } else {
      throw std::invalid_argument("unknown argument: " + argument);
    }
  }
  return parsed;
}

auto csv_field(const std::string &value) -> std::string {
  std::string output = "\"";
  for (const auto character : value) {
    if (character == '"')
      output += '"';
    output += character;
  }
  output += '"';
  return output;
}

auto digest_string(std::uint64_t digest) -> std::string {
  std::ostringstream output;
  output << std::hex << std::setw(16) << std::setfill('0') << digest;
  return output.str();
}

auto validate_result(const case_spec &spec, const benchmark_case &benchmark,
                     const result_summary &result) -> void {
  if (result.shape != benchmark.expected_shape ||
      result.length != benchmark.expected_length) {
    throw std::runtime_error(
        spec.name + ": expected output " + benchmark.expected_shape + " (" +
        std::to_string(benchmark.expected_length) + "), got " + result.shape +
        " (" + std::to_string(result.length) + ")");
  }
}

auto median(std::vector<double> values) -> double {
  std::sort(values.begin(), values.end());
  const auto middle = values.size() / 2;
  if (values.size() % 2 != 0)
    return values[middle];
  return (values[middle - 1] + values[middle]) * 0.5;
}

auto time_batch(const benchmark_case &benchmark, std::size_t operations)
    -> double {
  const auto start = clock_type::now();
  benchmark.timed_run(operations);
  const auto stop = clock_type::now();
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

auto calibrate_operations(const benchmark_case &benchmark) -> std::size_t {
  auto operations = std::size_t{1};
  while (true) {
    const auto elapsed = time_batch(benchmark, operations);
    if (elapsed >= minimum_sample_milliseconds ||
        operations == maximum_batch_operations)
      return operations;

    auto factor = std::size_t{128};
    if (elapsed > 0.0) {
      factor = static_cast<std::size_t>(
          std::ceil(minimum_sample_milliseconds / elapsed));
      factor = std::max(std::size_t{2}, std::min(std::size_t{128}, factor));
    }
    if (factor > maximum_batch_operations / operations)
      operations = maximum_batch_operations;
    else
      operations *= factor;
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto parsed = parse_options(argc, argv);
    const auto cases = make_cases();
    if (parsed.list) {
      for (const auto &spec : cases)
        if (parsed.filter.empty() ||
            spec.name.find(parsed.filter) != std::string::npos)
          std::cout << spec.name << '\n';
      return 0;
    }

    const auto tbb_available_concurrency =
        tbb::this_task_arena::max_concurrency();
    const auto tbb_default_concurrency = tbb::info::default_concurrency();

    std::cout
        << "case,dtype,parameters,output_shape,output_length,thread_cap,"
           "tbb_available_concurrency,tbb_default_concurrency,"
           "tbb_active_parallelism,samples,operations_per_sample,"
           "median_ms_per_operation,min_ms_per_operation,median_sample_ms,"
           "min_sample_ms,equivalence_digest,build_type,compiler,ipo,"
           "allocator\n";
    auto selected_count = 0;
    for (const auto &spec : cases) {
      if (!parsed.filter.empty() &&
          spec.name.find(parsed.filter) == std::string::npos)
        continue;
      ++selected_count;

      auto benchmark = spec.setup();
      auto reference_digest = std::uint64_t{0};
      auto have_reference_digest = false;
      std::vector<int> thread_caps{1};
      if (parsed.thread_cap != 1)
        thread_caps.push_back(parsed.thread_cap);
      for (const auto thread_cap : thread_caps) {
        tbb::global_control control(
            tbb::global_control::max_allowed_parallelism,
            static_cast<std::size_t>(thread_cap));
        const auto tbb_active_parallelism = tbb::global_control::active_value(
            tbb::global_control::max_allowed_parallelism);

        const auto warmup = benchmark.validation_run();
        validate_result(spec, benchmark, warmup);
        const auto consistency_check = benchmark.validation_run();
        validate_result(spec, benchmark, consistency_check);
        // Digests only check repeated-run determinism and byte equivalence
        // across thread caps. Focused C++ tests remain the semantic oracle.
        if (consistency_check.digest != warmup.digest)
          throw std::runtime_error(
              spec.name + ": repeated warmups produced different digests");
        if (have_reference_digest && warmup.digest != reference_digest)
          throw std::runtime_error(spec.name +
                                   ": thread-cap equivalence digests differ");
        reference_digest = warmup.digest;
        have_reference_digest = true;

        const auto operations = calibrate_operations(benchmark);
        std::vector<double> sample_timings;
        std::vector<double> operation_timings;
        sample_timings.reserve(static_cast<std::size_t>(parsed.samples));
        operation_timings.reserve(static_cast<std::size_t>(parsed.samples));
        for (int sample = 0; sample < parsed.samples; ++sample) {
          const auto sample_milliseconds = time_batch(benchmark, operations);
          sample_timings.push_back(sample_milliseconds);
          operation_timings.push_back(sample_milliseconds /
                                      static_cast<double>(operations));
        }

        const auto minimum_sample =
            *std::min_element(sample_timings.begin(), sample_timings.end());
        const auto minimum_operation = *std::min_element(
            operation_timings.begin(), operation_timings.end());
        std::cout << csv_field(spec.name) << ',' << csv_field(spec.dtype) << ','
                  << csv_field(spec.parameters) << ','
                  << csv_field(warmup.shape) << ',' << warmup.length << ','
                  << thread_cap << ',' << tbb_available_concurrency << ','
                  << tbb_default_concurrency << ',' << tbb_active_parallelism
                  << ',' << parsed.samples << ',' << operations << ','
                  << std::fixed << std::setprecision(6)
                  << median(operation_timings) << ',' << minimum_operation
                  << ',' << median(sample_timings) << ',' << minimum_sample
                  << ',' << digest_string(warmup.digest) << ','
                  << csv_field(TF_BENCHMARK_BUILD_TYPE) << ','
                  << csv_field(TF_BENCHMARK_COMPILER) << ','
                  << csv_field(TF_BENCHMARK_IPO) << ','
                  << csv_field(TF_BENCHMARK_ALLOCATOR) << '\n';
      }
    }

    if (selected_count == 0)
      throw std::runtime_error("case filter selected no benchmarks");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "benchmark error: " << error.what() << '\n';
    return 1;
  }
}
