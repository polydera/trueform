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
#include "../checked.hpp"
#include "../zip_apply.hpp"
#include "./block_reduce.hpp"
#include <type_traits>
#include <utility>

namespace tf {

/// @ingroup core_algorithms
/// @brief Generate variable-length output from each input element in parallel.
///
/// Processes each element of the input range and allows the generator
/// to push variable amounts of data to the output buffer. Block-local buffers
/// are used internally and merged after parallel processing.
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
/// @brief Generate with reusable block-local state.
///
/// Provides one state object to each sequential work block, useful for avoiding
/// repeated allocation inside the block's per-element loop.
template <typename Range, typename T, typename F, typename State>
auto generic_generate(const Range &r, tf::buffer<T> &buffer, const F &generator,
                      State local_state) {
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
/// @brief Generate into multiple buffers; the tail is the reusable
///        block-local state or tf::checked.
///
/// One function, if constexpr on the tail: MSVC cannot partially order
/// sibling overloads at this arity beside the buffer pack.
template <typename Range, typename... Ts, typename F, typename StateOrChecked>
auto generic_generate(const Range &r, std::tuple<tf::buffer<Ts> &...> buffers,
                      const F &generator, StateOrChecked tail) {
  if constexpr (std::is_same_v<StateOrChecked, tf::checked_t>) {
    if (std::size_t(r.size()) < tail.serial_below) {
      std::apply(
          [&](auto &...buffer) {
            (buffer.reserve(buffer.size() + r.size()), ...);
          },
          buffers);
      for (const auto &element : r)
        generator(element, buffers);
    } else
      generic_generate(r, buffers, generator);
  } else {
    tf::blocked_reduce(
        r, buffers, std::make_pair(std::tuple<tf::buffer<Ts>...>{}, tail),
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
}
/// @ingroup core_algorithms
/// @brief Checked generation: sequential below the parallel entry cost.
///
/// The serial path appends straight into the output — no block-local
/// buffers, no merge copy. Aggregate order is unspecified either way,
/// so the fallback changes nothing a caller may rely on.
template <typename Range, typename T, typename F>
auto generic_generate(const Range &r, tf::buffer<T> &buffer,
                      const F &generator, tf::checked_t c) {
  if (std::size_t(r.size()) < c.serial_below) {
    buffer.reserve(buffer.size() + r.size());
    for (const auto &element : r)
      generator(element, buffer);
  } else
    generic_generate(r, buffer, generator);
}

/// @ingroup core_algorithms
/// @brief Checked generation with reusable block-local state.
///
/// The serial path is a single block, so one state carries the whole range.
template <typename Range, typename T, typename F, typename State>
auto generic_generate(const Range &r, tf::buffer<T> &buffer, const F &generator,
                      State local_state, tf::checked_t c) {
  if (std::size_t(r.size()) < c.serial_below) {
    buffer.reserve(buffer.size() + r.size());
    for (const auto &element : r)
      generator(element, buffer, local_state);
  } else
    generic_generate(r, buffer, generator, std::move(local_state));
}

/// @ingroup core_algorithms
/// @brief Checked multi-buffer generation with reusable block-local state.
template <typename Range, typename... Ts, typename F, typename State>
auto generic_generate(const Range &r, std::tuple<tf::buffer<Ts> &...> buffers,
                      const F &generator, State local_state, tf::checked_t c) {
  if (std::size_t(r.size()) < c.serial_below) {
    std::apply(
        [&](auto &...buffer) {
          (buffer.reserve(buffer.size() + r.size()), ...);
        },
        buffers);
    for (const auto &element : r)
      generator(element, buffers, local_state);
  } else
    generic_generate(r, buffers, generator, std::move(local_state));
}
} // namespace tf
