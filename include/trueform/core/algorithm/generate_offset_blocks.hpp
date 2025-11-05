/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../blocked_buffer.hpp"
#include "../offset_block_buffer.hpp"
#include "./block_reduce_sequenced_aggregate.hpp"
namespace tf {
namespace core {
template <typename Range, typename Index, typename Buffer, typename F>
auto generate_offset_blocks(
    const Range &input_data, tf::buffer<Index> &offsets, Buffer &data,
    const F &fill_block_f,
    std::size_t n_tasks = std::thread::hardware_concurrency() * 5) {
  if (!input_data.size())
    return;
  offsets.allocate(input_data.size() + 1);
  offsets[0] = 0;
  std::size_t current_i = 1;
  auto task_f = [fill_block_f](auto &&range,
                               std::pair<tf::buffer<Index>, Buffer> &pair) {
    auto &[sizes, data] = pair;
    sizes.allocate(range.size());
    data.reserve(data.size() * 5);
    auto it = sizes.begin();
    for (const auto &element : range) {
      auto old_size = data.size();
      fill_block_f(element, data);
      *it++ = data.size() - old_size;
    }
  };
  auto aggregate_f =
      [&](const std::pair<tf::buffer<Index>, Buffer> &local_result,
                   std::tuple<tf::buffer<Index> &, Buffer &> &result) {
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
  tf::blocked_reduce_sequenced_aggregate(input_data, std::tie(offsets, data),
                                         std::pair<tf::buffer<Index>, Buffer>{},
                                         task_f, aggregate_f, n_tasks);
}
} // namespace core

template <typename Range, typename Index, typename T, typename F>
auto generate_offset_blocks(
    const Range &input_data, tf::buffer<Index> &offsets, tf::buffer<T> &data,
    const F &fill_block_f,
    std::size_t n_tasks = std::thread::hardware_concurrency() * 5) {
  return core::generate_offset_blocks(input_data, offsets, data, fill_block_f,
                                      n_tasks);
}

template <typename Range, typename Index, typename T, std::size_t N, typename F>
auto generate_offset_blocks(
    const Range &input_data, tf::buffer<Index> &offsets,
    tf::blocked_buffer<T, N> &data, const F &fill_block_f,
    std::size_t n_tasks = std::thread::hardware_concurrency() * 5) {
  return core::generate_offset_blocks(input_data, offsets, data, fill_block_f,
                                      n_tasks);
}

template <typename Range, typename Index, typename T, typename F>
auto generate_offset_blocks(
    const Range &input_data, tf::offset_block_buffer<Index, T> &buff,
    const F &fill_block_f,
    std::size_t n_tasks = std::thread::hardware_concurrency() * 5) {
  generate_offset_blocks(input_data, buff.offsets_buffer(), buff.data_buffer(),
                         fill_block_f, n_tasks);
}
} // namespace tf
