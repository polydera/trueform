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

#include "../../../core/buffer.hpp"

namespace tf::exact::door::pool {

/// What each ORIGINAL VERTEX of a scene aims at: the name whose committed
/// plane its placement reads, and whether it is a feature.
///
/// A feature has a target source like any other; what retires it is that the
/// rank-3/2 cascade owns its position, and this tier's placement leaves it
/// where it stands.
struct scene_targets {
  tf::buffer<int> source;
  tf::buffer<char> feature;
};

} // namespace tf::exact::door::pool
