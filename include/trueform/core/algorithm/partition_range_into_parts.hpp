#pragma once
#include "../range.hpp"
#include "tbb/parallel_invoke.h"

namespace tf {
/// @ingroup core_algorithms
/// @brief Recursively partitions a range into a given number of parts and
/// applies a user function to each part.
///
/// This utility enables balanced, recursive partitioning of a range into
/// `parts` disjoint subranges, optionally executing the process in parallel. It
/// is typically used to build hierarchical spatial structures (e.g., BVHs or
/// spatial trees), where you want to partition a data range and apply a
/// function (e.g., to assign nodes or record split information) to each
/// partition.
///
/// The user supplies:
/// - `partition_range_f`: a function that reorders or partitions elements in
/// `[begin, end)`
///                        so that the middle point `mid` creates two logical
///                        subranges.
/// - `apply_f`: a function that will be applied to each resulting range after
/// partitioning, with its partition id.
///
/// If the range is small (≤ 1000 elements), it runs sequentially; otherwise it
/// uses `tbb::parallel_invoke`.
///
/// @tparam Range The input range type (should support `.begin()`, `.end()`, and
/// `operator-`).
/// @tparam F0 Type of the partitioning function. Should have signature
/// `(Iterator begin, Iterator mid, Iterator end)`.
/// @tparam F1 Type of the apply function. Should have signature `(Range
/// subrange, std::size_t partition_id)`.
/// @param range The full range to be partitioned.
/// @param parts The number of parts to divide the range into.
/// @param partition_range_f A function to partition the range at midpoint.
/// @param apply_f A function to apply to each resulting subrange.
/// @param partition_id Internal use (default 0). Tracks the base index of the
/// current partition.
/// @return The total number of successfully partitioned subranges (should equal
/// `parts`).
///
/// @code
/// // Partition a vector into 4 parts, sorting each part independently.
/// tf::buffer<int> data = {8, 1, 5, 2, 9, 7, 4, 3};
/// tf::partition_range_into_parts(
///     tf::make_range(data),
///     4,
///     [](auto begin, auto mid, auto end) {
///         std::nth_element(begin, mid, end);
///     },
///     [](auto range, std::size_t id) {
///         std::sort(range.begin(), range.end());
///         std::cout << "Partition " << id << ": ";
///         for (auto v : range) std::cout << v << ' ';
///         std::cout << '\n';
///     });
/// @endcode
template <typename Range, typename F0, typename F1>
auto partition_range_into_parts(Range &&range, std::size_t parts,
                                const F0 &partition_range_f, const F1 &apply_f,
                                std::size_t partition_id = 0) -> std::size_t {
  size_t total_size = range.end() - range.begin();
  parts = std::min(parts, static_cast<std::size_t>(total_size));

  if (parts == 1) {
    apply_f(range, partition_id);
    return 1;
  }

  std::size_t left_parts = parts / 2;
  std::size_t right_parts = parts - left_parts;

  size_t mid_size = total_size * left_parts / parts;
  auto mid = range.begin() + mid_size;

  partition_range_f(range.begin(), mid, range.end());

  std::size_t count_left = 0;
  std::size_t count_right = 0;

  if (range.size() > 1000) {
    tbb::parallel_invoke(
        [&]() {
          count_left = partition_range_into_parts(
              tf::make_range(range.begin(), mid), left_parts, partition_range_f,
              apply_f, partition_id);
        },
        [&]() {
          count_right = partition_range_into_parts(
              tf::make_range(mid, range.end()), right_parts, partition_range_f,
              apply_f, partition_id + left_parts);
        });
  } else {
    count_left = partition_range_into_parts(tf::make_range(range.begin(), mid),
                                            left_parts, partition_range_f,
                                            apply_f, partition_id);
    count_right = partition_range_into_parts(
        tf::make_range(mid, range.end()), right_parts, partition_range_f,
        apply_f, partition_id + left_parts);
  }

  return count_left + count_right;
}
} // namespace tf
