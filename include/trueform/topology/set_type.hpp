/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
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
