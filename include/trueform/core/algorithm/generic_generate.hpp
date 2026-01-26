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
#pragma once
#include "../buffer.hpp"
#include "../zip_apply.hpp"
#include "./block_reduce.hpp"

namespace tf {

/// @ingroup core_algorithms
/// @brief Generate variable-length output from each input element in parallel.
///
/// Processes each element of the input range and allows the generator
/// to push variable amounts of data to the output buffer. Thread-local
/// buffers are used internally and merged after parallel processing.
///
/// @tparam Range The input range type.
/// @tparam T The output buffer element type.
/// @tparam F Generator function: `void(const element&, tf::buffer<T>&)`.
/// @param r The input range.
/// @param buffer The output buffer (appended to).
/// @param generator Function that processes an element and pushes results.
template <typename Range, typename T, typename F>
auto generic_generate(const Range &r, tf::buffer<T> &buffer,
                      const F &generator) {
  tf::blocked_reduce(
      r, buffer, tf::buffer<T>{},
      [generator](const auto &r, auto &buffer) {
        buffer.reserve(r.size());
        for (const auto &element : r)
          generator(element, buffer);
      },
      [](const auto &local_buffer, auto &buffer) {
        auto old_size = buffer.size();
        buffer.reallocate(old_size + local_buffer.size());
        std::copy(local_buffer.begin(), local_buffer.end(),
                  buffer.begin() + old_size);
      });
}

/// @ingroup core_algorithms
/// @brief Generate with thread-local state for each block.
///
/// Provides a thread-local state object to each worker, useful for
/// avoiding repeated allocations of temporary work buffers.
template <typename Range, typename T, typename State, typename F>
auto generic_generate(const Range &r, tf::buffer<T> &buffer, State local_state,
                      const F &generator) {
  tf::blocked_reduce(
      r, buffer, std::make_pair(tf::buffer<T>{}, local_state),
      [generator](const auto &r, auto &pair) {
        auto &[buffer, local_state] = pair;
        buffer.reserve(r.size());
        for (const auto &element : r)
          generator(element, buffer, local_state);
      },
      [](const auto &pair, auto &buffer) {
        auto &local_buffer = pair.first;
        auto old_size = buffer.size();
        buffer.reallocate(old_size + local_buffer.size());
        std::copy(local_buffer.begin(), local_buffer.end(),
                  buffer.begin() + old_size);
      });
}

/// @ingroup core_algorithms
/// @brief Generate into multiple output buffers simultaneously.
template <typename Range, typename... Ts, typename F>
auto generic_generate(const Range &r, std::tuple<tf::buffer<Ts> &...> buffers,
                      const F &generator) {
  tf::blocked_reduce(
      r, buffers, std::tuple<tf::buffer<Ts>...>{},
      [generator](const auto &r, auto &buffers) {
        std::apply([&](auto &...buffer) { (buffer.reserve(r.size()), ...); },
                   buffers);
        for (const auto &element : r)
          generator(element, buffers);
      },
      [](const auto &local_buffer, auto &buffer) {
        tf::zip_apply(
            [](auto &&...tups) {
              (
                  [](auto &&tup) {
                    auto &&[local_buffer, buffer] = tup;
                    auto old_size = buffer.size();
                    buffer.reallocate(old_size + local_buffer.size());
                    std::copy(local_buffer.begin(), local_buffer.end(),
                              buffer.begin() + old_size);
                  }(tups),
                  ...);
            },
            local_buffer, buffer);
      });
}

/// @ingroup core_algorithms
/// @brief Generate into multiple buffers with thread-local state.
template <typename Range, typename... Ts, typename State, typename F>
auto generic_generate(const Range &r, std::tuple<tf::buffer<Ts> &...> buffers,
                      State local_state, const F &generator) {
  tf::blocked_reduce(
      r, buffers, std::make_pair(std::tuple<tf::buffer<Ts>...>{}, local_state),
      [generator](const auto &r, auto &pair) {
        auto &[buffers, local_state] = pair;
        std::apply([&](auto &...buffer) { (buffer.reserve(r.size()), ...); },
                   buffers);
        for (const auto &element : r)
          generator(element, buffers, local_state);
      },
      [](const auto &pair, auto &buffer) {
        auto &local_buffer = pair.first;
        tf::zip_apply(
            [](auto &&...tups) {
              (
                  [](auto &&tup) {
                    auto &&[local_buffer, buffer] = tup;
                    auto old_size = buffer.size();
                    buffer.reallocate(old_size + local_buffer.size());
                    std::copy(local_buffer.begin(), local_buffer.end(),
                              buffer.begin() + old_size);
                  }(tups),
                  ...);
            },
            local_buffer, buffer);
      });
}
} // namespace tf
