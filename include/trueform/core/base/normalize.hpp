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
* Author: Ziga Sajovic
*/
#pragma once
#include "../vector_like.hpp"

namespace tf::core {
template <std::size_t N, typename T>
auto normalize(vector_like<N, T> &v) -> vector_like<N, T> & {
  auto d = v.length();
  v /= d + (d == 0);
  return v;
}
} // namespace tf::core
