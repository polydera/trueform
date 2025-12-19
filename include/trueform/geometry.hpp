/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./geometry/chamfer_error.hpp"               // IWYU pragma: export
#include "./geometry/compute_normals.hpp"             // IWYU pragma: export
#include "./geometry/compute_point_normals.hpp"       // IWYU pragma: export
#include "./geometry/ensure_positive_orientation.hpp" // IWYU pragma: export
#include "./geometry/fit_knn_alignment.hpp"           // IWYU pragma: export
#include "./geometry/fit_obb_alignment.hpp"           // IWYU pragma: export
#include "./geometry/fit_rigid_alignment.hpp"         // IWYU pragma: export
#include "./geometry/fit_similarity_alignment.hpp"    // IWYU pragma: export
#include "./geometry/impl/ear_cutter.hpp"             // IWYU pragma: export
#include "./geometry/laplacian_smoothed.hpp"          // IWYU pragma: export
#include "./geometry/make_sphere_mesh.hpp"            // IWYU pragma: export
#include "./geometry/triangulated.hpp"                // IWYU pragma: export
#include "./geometry/triangulated_faces.hpp"          // IWYU pragma: export
