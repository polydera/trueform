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
  meshed,  ///< Every constraint recovered and the surface winds
           ///< consistently; the mesh is the cells it winds positively.
           ///< A surface of no face is meshed by no cell.
  refused, ///< Some constraint was refused, the entrance left a
           ///< carrier without its triangles, or its lattice cannot
           ///< resolve the crossings its passes kept naming, which
           ///< the caller answers by stating the deeper lattice,
           ///< `tf::make_tet_mesh<std::int64_t>`. Each kind is typed
           ///< and counted in the report; the mesh is emitted only
           ///< when the surface still winds consistently.
  unwound, ///< The surface does not wind consistently — an
           ///< inconsistently wound input or one with sheets, a flap's
           ///< free rim among them — read on the recovered surface, or by
           ///< the region labeling, after every constraint was recovered;
           ///< no mesh is emitted.
  rank_deficient, ///< The input's points span less than three dimensions.
  capacity,       ///< The cells or sites outgrew the index type.
  open            ///< The recovered surface has a hole: a rim of free
                  ///< edges that hangs off no non-manifold attachment
                  ///< (@ref tf::has_holes); no mesh is emitted.
                  ///< @ref tf::fill_holes closes a hole.
};

} // namespace tf
