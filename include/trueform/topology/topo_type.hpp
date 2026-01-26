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
/// @brief Identifies the type of topological element.
///
/// Used to distinguish between different mesh elements (vertices, edges, faces)
/// in generic topology algorithms. Values are powers of 2 to allow combining
/// as flags.
enum class topo_type : char {
  none = 0,    ///< No element type.
  vertex = 1,  ///< A vertex (0-dimensional).
  edge = 2,    ///< An edge (1-dimensional).
  face = 4     ///< A face (2-dimensional).
};
}
