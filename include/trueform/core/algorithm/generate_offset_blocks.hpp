/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./block_reduce_sequenced_aggregate.hpp"
#include "../offset_block_buffer.hpp"
namespace tf {
template <typename Range, typename Index, typename T, typename F>
auto generate_offset_blocks(const Range &input_data, tf::buffer<Index> &offsets,
                            tf::buffer<T> &data, const F &fill_block_f) {
  if (!input_data.size())
    return;
  offsets.allocate(input_data.size() + 1);
  offsets[0] = 0;
  std::size_t current_i = 1;
  auto task_f =
      [fill_block_f](auto &&range,
                     std::pair<tf::buffer<Index>, tf::buffer<T>> &pair) {
        auto &[sizes, data] = pair;
        sizes.allocate(range.size());
        data.reserve(data.size() * 5);
        auto it = sizes.begin();
        for (const auto &element : range) {
          fill_block_f(element, data);
          *it++ = data.size();
        }
      };
  auto aggregate_f =
      [&current_i](
          const std::pair<tf::buffer<Index>, tf::buffer<T>> &local_result,
          std::tuple<tf::buffer<Index> &, tf::buffer<T> &> &result) {
        auto &[l_sizes, l_data] = local_result;
        auto &[offsets, data] = result;
        auto old_data_size = data.size();
        data.reallocate(old_data_size + l_data.size());
        std::copy(l_data.begin(), l_data.end(), data.begin() + old_data_size);
        for (auto offset : l_sizes) {
          offsets[current_i] = offsets[current_i - 1] + offset;
          current_i++;
        }
      };
  tf::blocked_reduce_sequenced_aggregate(
      input_data, std::tie(offsets, data),
      std::pair<tf::buffer<Index>, tf::buffer<T>>{}, task_f, aggregate_f);
}

template <typename Range, typename Index, typename T, typename F>
auto generate_offset_blocks(const Range &input_data,
                            tf::offset_block_buffer<Index, T> &buff,
                            const F &fill_block_f) {
  generate_offset_blocks(input_data, buff.offsets_buffer(), buff.data_buffer(),
                         fill_block_f);
}
} // namespace tf
