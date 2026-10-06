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
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// A site's process key: the splitmix64 mix of its canonical name. The mix
/// is a bijection, so distinct names carry distinct keys and the name itself
/// only states the total order.
inline auto tet_site_priority(std::uint64_t name) -> std::uint64_t {
  std::uint64_t x = name ^ 0x9E3779B97F4A7C15ull;
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

} // namespace tf::topology::cdt::dt3
