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

/** @defgroup cpp_geometry geometry
 *  Measurements, normals, curvature, smoothing, primitives.
 */

#include "./geometry/async/area.hpp"                    // IWYU pragma: export
#include "./geometry/async/chamfer_error.hpp"           // IWYU pragma: export
#include "./geometry/async/dihedral_angles.hpp"         // IWYU pragma: export
#include "./geometry/async/face_quality.hpp"            // IWYU pragma: export
#include "./geometry/async/fit_icp.hpp"                 // IWYU pragma: export
#include "./geometry/async/fit_knn.hpp"                 // IWYU pragma: export
#include "./geometry/async/fit_obb.hpp"                 // IWYU pragma: export
#include "./geometry/async/fit_rigid.hpp"               // IWYU pragma: export
#include "./geometry/async/laplacian_smoothed.hpp"      // IWYU pragma: export
#include "./geometry/async/make_box_mesh.hpp"           // IWYU pragma: export
#include "./geometry/async/make_cylinder_mesh.hpp"      // IWYU pragma: export
#include "./geometry/async/make_plane_mesh.hpp"         // IWYU pragma: export
#include "./geometry/async/make_sphere_mesh.hpp"        // IWYU pragma: export
#include "./geometry/async/make_tube_mesh.hpp"          // IWYU pragma: export
#include "./geometry/async/max_edge_length.hpp"         // IWYU pragma: export
#include "./geometry/async/mean_edge_length.hpp"        // IWYU pragma: export
#include "./geometry/async/min_edge_length.hpp"         // IWYU pragma: export
#include "./geometry/async/normals.hpp"                 // IWYU pragma: export
#include "./geometry/async/point_normals.hpp"           // IWYU pragma: export
#include "./geometry/async/positively_oriented.hpp"     // IWYU pragma: export
#include "./geometry/async/principal_curvatures.hpp"    // IWYU pragma: export
#include "./geometry/async/principal_directions.hpp"    // IWYU pragma: export
#include "./geometry/async/reverse_winding.hpp"         // IWYU pragma: export
#include "./geometry/async/shape_index.hpp"             // IWYU pragma: export
#include "./geometry/async/sharp_edges.hpp"             // IWYU pragma: export
#include "./geometry/async/signed_volume.hpp"           // IWYU pragma: export
#include "./geometry/async/symmetric_chamfer_error.hpp" // IWYU pragma: export
#include "./geometry/async/taubin_smoothed.hpp"         // IWYU pragma: export
#include "./geometry/async/triangulate.hpp"             // IWYU pragma: export
#include "./geometry/async/volume.hpp"                  // IWYU pragma: export
#include "./geometry/area.hpp"                    // IWYU pragma: export
#include "./geometry/chamfer_error.hpp"           // IWYU pragma: export
#include "./geometry/chamfer_error_options.hpp"   // IWYU pragma: export
#include "./geometry/dihedral_angles.hpp"         // IWYU pragma: export
#include "./geometry/face_quality.hpp"            // IWYU pragma: export
#include "./geometry/fit_icp.hpp"                 // IWYU pragma: export
#include "./geometry/fit_knn.hpp"                 // IWYU pragma: export
#include "./geometry/fit_obb.hpp"                 // IWYU pragma: export
#include "./geometry/fit_rigid.hpp"               // IWYU pragma: export
#include "./geometry/laplacian_smoothed.hpp"      // IWYU pragma: export
#include "./geometry/make_box_mesh.hpp"           // IWYU pragma: export
#include "./geometry/make_cylinder_mesh.hpp"      // IWYU pragma: export
#include "./geometry/make_plane_mesh.hpp"         // IWYU pragma: export
#include "./geometry/make_sphere_mesh.hpp"        // IWYU pragma: export
#include "./geometry/make_tube_mesh.hpp"          // IWYU pragma: export
#include "./geometry/max_edge_length.hpp"         // IWYU pragma: export
#include "./geometry/mean_edge_length.hpp"        // IWYU pragma: export
#include "./geometry/min_edge_length.hpp"         // IWYU pragma: export
#include "./geometry/normals.hpp"                 // IWYU pragma: export
#include "./geometry/point_normals.hpp"           // IWYU pragma: export
#include "./geometry/positively_oriented.hpp"     // IWYU pragma: export
#include "./geometry/principal_curvatures.hpp"    // IWYU pragma: export
#include "./geometry/principal_directions.hpp"    // IWYU pragma: export
#include "./geometry/reverse_winding.hpp"         // IWYU pragma: export
#include "./geometry/shape_index.hpp"             // IWYU pragma: export
#include "./geometry/sharp_edges.hpp"             // IWYU pragma: export
#include "./geometry/signed_volume.hpp"           // IWYU pragma: export
#include "./geometry/symmetric_chamfer_error.hpp" // IWYU pragma: export
#include "./geometry/taubin_smoothed.hpp"         // IWYU pragma: export
#include "./geometry/triangulate.hpp"             // IWYU pragma: export
#include "./geometry/volume.hpp"                  // IWYU pragma: export
