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

/// @ingroup topology
/// @brief Which triangulation a consumer builds over cut surfaces.
///
/// `cdt` is the plain constrained Delaunay triangulation of each cut
/// loop; `refined_cdt` additionally quality-refines the cut surface
/// (Ruppert circumcenter insertion with negotiated boundary splits), so
/// shared loop boundaries stay watertight by construction.
enum class triangulation_type { cdt = 0, refined_cdt = 1 };

} // namespace tf
