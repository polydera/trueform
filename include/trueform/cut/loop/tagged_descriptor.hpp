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
#include <utility>
namespace tf::loop {
template <typename Index> struct tagged_descriptor {
  Index tag;
  Index object;

  tagged_descriptor() = default;
  tagged_descriptor(std::pair<Index, Index> p)
      : tag{p.first}, object{p.second} {}
};
} // namespace tf::loop
