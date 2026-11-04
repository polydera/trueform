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

namespace tf {

/// What @ref tf::make_tet_mesh made of its input.
enum class tet_mesh_status : std::uint8_t {
  meshed,         ///< Every constraint recovered and the surface winds
                  ///< consistently; the mesh is the cells it winds positively.
  refused,        ///< Some constraint was refused, or the entrance left a
                  ///< carrier without its triangles, each typed and counted
                  ///< in the report; the mesh is emitted only when the
                  ///< surface still winds consistently.
  unwound,        ///< The surface does not wind consistently — an
                  ///< inconsistently wound input or one with sheets — read at
                  ///< the entrance, or by the carve after every constraint was
                  ///< recovered; no mesh is emitted.
  rank_deficient, ///< The input's points span less than three dimensions.
  capacity,       ///< The cells or sites outgrew the index type.
  open            ///< The input's surface has a hole: an edge of it one
                  ///< triangle alone holds; nothing is built and no mesh is
                  ///< emitted.
};

} // namespace tf
