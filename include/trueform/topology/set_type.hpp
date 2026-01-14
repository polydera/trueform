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
/// @brief Specifies whether a mesh or region is topologically open or closed.
///
/// A closed mesh has no boundary edges (every edge is shared by exactly two
/// faces), while an open mesh has boundary edges.
enum class set_type : char {
  open,   ///< The mesh has boundary edges.
  closed  ///< The mesh has no boundary edges.
};
}
