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

/** @defgroup geometry Geometry Module
 *  Mesh generation, normal computation, processing, and registration.
 */

/** @defgroup geometry_meshing Mesh Generation
 *  @ingroup geometry
 *  Generate primitive meshes (sphere, cylinder, box, plane).
 */

/** @defgroup geometry_normals Normal Computation
 *  @ingroup geometry
 *  Compute face normals and vertex normals.
 */

/** @defgroup geometry_processing Mesh Processing
 *  @ingroup geometry
 *  Triangulation, smoothing, and orientation operations.
 */

/** @defgroup geometry_registration Point Cloud Registration
 *  @ingroup geometry
 *  Alignment and error metrics for point clouds.
 */

#include "./geometry/chamfer_error.hpp"               // IWYU pragma: export
#include "./geometry/compute_normals.hpp"             // IWYU pragma: export
#include "./geometry/compute_point_normals.hpp"       // IWYU pragma: export
#include "./geometry/compute_principal_curvatures.hpp" // IWYU pragma: export
#include "./geometry/compute_shape_index.hpp"         // IWYU pragma: export
#include "./geometry/ensure_positive_orientation.hpp" // IWYU pragma: export
#include "./geometry/fit_knn_alignment.hpp"           // IWYU pragma: export
#include "./geometry/fit_obb_alignment.hpp"           // IWYU pragma: export
#include "./geometry/fit_rigid_alignment.hpp"         // IWYU pragma: export
#include "./geometry/fit_similarity_alignment.hpp"    // IWYU pragma: export
#include "./geometry/impl/ear_cutter.hpp"             // IWYU pragma: export
#include "./geometry/laplacian_smoothed.hpp"          // IWYU pragma: export
#include "./geometry/make_box_mesh.hpp"               // IWYU pragma: export
#include "./geometry/make_cylinder_mesh.hpp"          // IWYU pragma: export
#include "./geometry/make_plane_mesh.hpp"             // IWYU pragma: export
#include "./geometry/make_sphere_mesh.hpp"            // IWYU pragma: export
#include "./geometry/triangulated.hpp"                // IWYU pragma: export
#include "./geometry/triangulated_faces.hpp"          // IWYU pragma: export
