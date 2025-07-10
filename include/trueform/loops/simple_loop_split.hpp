/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/buffer.hpp"
#include <algorithm>

namespace tf {
template <typename Range, typename Index, typename T>
auto simple_loop_split(const Range &base_loop, std::array<T, 2> edge,
                       tf::buffer<Index> &offsets, tf::buffer<T> &vertices) {
  auto buffer_start_offset = vertices.size();
  auto it = base_loop.begin();
  auto end = base_loop.end();
  offsets.push_back(vertices.size());
  while (it != end) {
    vertices.push_back(*it);
    if (*it == edge[1])
      std::swap(edge[0], edge[1]);
    if (*it == edge[0]) {
      auto buffer_offset = vertices.size();
      while (*it != edge[1]) {
        vertices.push_back(*it++);
      }
      vertices.push_back(edge[1]);
      std::rotate(vertices.begin() + buffer_start_offset,
                  vertices.begin() + buffer_offset, vertices.end());
      offsets.push_back(offsets.back() + vertices.size() - buffer_offset);
      vertices.push_back(edge[1]);
    }
    it++;
  }
}
} // namespace tf
