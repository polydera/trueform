/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <utility>
namespace tf::loop {
template <typename Index> struct descriptor {
  Index tag;
  Index object;

  descriptor() = default;
  descriptor(std::pair<Index, Index> p) : tag{p.first}, object{p.second} {}
};
} // namespace tf::loop
