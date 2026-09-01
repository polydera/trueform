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
#include <cstdint>
#include <type_traits>

namespace tf::topology::cdt {

template <typename Int> struct delaunay_site_domain {
  auto x_offset(Int x) const -> std::uint64_t {
    using UInt = std::make_unsigned_t<Int>;
    return std::uint64_t(UInt(x) - UInt(minimum_x));
  }

  auto y_offset(Int y) const -> std::uint64_t {
    using UInt = std::make_unsigned_t<Int>;
    return std::uint64_t(UInt(y) - UInt(minimum_y));
  }

  Int minimum_x;
  Int minimum_y;
  std::uint64_t span_x;
  std::uint64_t span_y;
};

} // namespace tf::topology::cdt
