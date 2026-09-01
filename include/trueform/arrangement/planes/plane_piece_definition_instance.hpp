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

#include <tuple>

namespace tf::arrangement {

template <typename Definition>
auto same_plane_piece_definition_instance(const Definition &x,
                                          const Definition &y) -> bool {
  return std::tie(x.face, x.tag_other, x.object_other, x.ordinal, x.side) ==
         std::tie(y.face, y.tag_other, y.object_other, y.ordinal, y.side);
}

} // namespace tf::arrangement
