/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../buffer.hpp"
#include "../zip_apply.hpp"
#include "./block_reduce.hpp"

namespace tf {
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
} // namespace tf
