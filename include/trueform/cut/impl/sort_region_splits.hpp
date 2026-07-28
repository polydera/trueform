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

#include "../../core/buffer.hpp"
#include <algorithm>

namespace tf::cut {

template <typename Split>
auto sort_region_splits(tf::buffer<Split> &splits) -> void {
  std::sort(splits.begin(), splits.end(),
            [](const Split &a, const Split &b) {
              if (a.edge != b.edge)
                return a.edge < b.edge;
              return a.param < b.param;
            });
  splits.erase_till_end(
      std::unique(splits.begin(), splits.end(),
                  [](const Split &a, const Split &b) {
                    return a.edge == b.edge && a.key == b.key;
                  }));
}

} // namespace tf::cut
