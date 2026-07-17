/*
 * Copyright (c) 2026 XLAB
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
#include "./block_reduce_sequenced_aggregate.hpp"

namespace tf {

/// @ingroup core_algorithms
/// @brief Generate variable-length output from each input element in
///        parallel, output in input order.
///
/// Same contract as @ref tf::generic_generate, but block results are
/// aggregated in block order, so the output concatenation is
/// deterministic — element `i`'s output always precedes element
/// `j > i`'s.
///
/// @tparam Range The input range type.
/// @tparam T The output buffer element type.
/// @tparam F Generator function: `void(const element&, tf::buffer<T>&)`.
/// @param r The input range.
/// @param buffer The output buffer (appended to).
/// @param generator Function that processes an element and pushes results.
template <typename Range, typename T, typename F>
auto sequenced_generate(const Range &r, tf::buffer<T> &buffer,
                        const F &generator) {
  tf::blocked_reduce_sequenced_aggregate(
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
/// @brief Ordered generate with thread-local state for each block.
///
/// Provides a thread-local state object to each worker, useful for
/// avoiding repeated allocations of temporary work buffers.
template <typename Range, typename T, typename State, typename F>
auto sequenced_generate(const Range &r, tf::buffer<T> &buffer,
                        State local_state, const F &generator) {
  tf::blocked_reduce_sequenced_aggregate(
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
/// @brief Ordered generate into multiple output buffers simultaneously.
template <typename Range, typename... Ts, typename F>
auto sequenced_generate(const Range &r, std::tuple<tf::buffer<Ts> &...> buffers,
                        const F &generator) {
  tf::blocked_reduce_sequenced_aggregate(
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
/// @brief Ordered generate into multiple buffers with thread-local state.
template <typename Range, typename... Ts, typename State, typename F>
auto sequenced_generate(const Range &r, std::tuple<tf::buffer<Ts> &...> buffers,
                        State local_state, const F &generator) {
  tf::blocked_reduce_sequenced_aggregate(
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
