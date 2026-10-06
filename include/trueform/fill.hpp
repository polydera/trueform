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

/** @defgroup fill Fill Module
 *  Hole filling: state a patch for every boundary rim of a mesh
 *  (@ref tf::fill_holes), then read the mesh those patches and the
 *  carrying faces they split make (@ref tf::make_filled_mesh).
 */

#include "./fill/fill_holes.hpp"        // IWYU pragma: export
#include "./fill/hole_fill_config.hpp"  // IWYU pragma: export
#include "./fill/hole_fill_result.hpp"  // IWYU pragma: export
#include "./fill/hole_fill_status.hpp"  // IWYU pragma: export
#include "./fill/hole_split.hpp"        // IWYU pragma: export
#include "./fill/make_filled_mesh.hpp"  // IWYU pragma: export
