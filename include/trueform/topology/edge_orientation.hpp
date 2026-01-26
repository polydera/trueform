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

namespace tf {

/// @ingroup topology_types
/// @brief Specifies how edges are oriented in connectivity structures.
///
/// Controls whether edge membership structures store edges in their
/// natural direction, reversed, or both directions.
enum class edge_orientation : signed char {
  forward = 0,       ///< Store edges in their natural direction (v0 → v1).
  reverse = 1,       ///< Store edges in reverse direction (v1 → v0).
  bidirectional = 3  ///< Store edges in both directions.
};
}
