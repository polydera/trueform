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
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt {

template <bool BuildIndexMap, typename Owner>
auto weld_delaunay_sites(Owner &owner) -> void {
  const auto site_order = [](const auto &a, const auto &b) {
    return a.x < b.x ||
           (a.x == b.x && (a.y < b.y || (a.y == b.y && a.output < b.output)));
  };

  std::size_t output = 0;
  for (std::size_t first = 0; first < owner._sites.size();) {
    std::size_t last = first + 1;
    while (last < owner._sites.size() &&
           owner._keys[last] == owner._keys[first])
      ++last;
    if (last - first > 1)
      std::sort(owner._sites.begin() + std::ptrdiff_t(first),
                owner._sites.begin() + std::ptrdiff_t(last), site_order);

    const auto key = owner._keys[first];
    auto kept_site = owner._sites[first];
    if (output != first) {
      owner._sites[output] = kept_site;
      owner._keys[output] = key;
    }
    ++output;

    for (std::size_t i = first + 1; i < last; ++i) {
      const auto current = owner._sites[i];
      if (current.x == kept_site.x && current.y == kept_site.y) {
        if constexpr (BuildIndexMap)
          owner._index_map.f()[std::size_t(current.output)] = kept_site.output;
        continue;
      }
      kept_site = current;
      if (output != i) {
        owner._sites[output] = current;
        owner._keys[output] = key;
      }
      ++output;
    }
    first = last;
  }
  if (output != owner._sites.size()) {
    owner._sites.allocate(output);
    owner._keys.allocate(output);
  }
}

} // namespace tf::topology::cdt
