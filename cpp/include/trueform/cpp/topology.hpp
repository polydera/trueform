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

/** @defgroup cpp_topology topology
 *  Connectivity, boundaries, components and domain labels.
 */

#include "./topology/async/cdt.hpp"                        // IWYU pragma: export
#include "./topology/async/cell_membership.hpp"            // IWYU pragma: export
#include "./topology/async/connect_edges_to_paths.hpp"     // IWYU pragma: export
#include "./topology/async/domain_labels.hpp"              // IWYU pragma: export
#include "./topology/async/face_link.hpp"                  // IWYU pragma: export
#include "./topology/async/k_rings.hpp"                    // IWYU pragma: export
#include "./topology/async/label_connected_components.hpp" // IWYU pragma: export
#include "./topology/async/manifold_edge_link.hpp"         // IWYU pragma: export
#include "./topology/async/mesh_queries.hpp"               // IWYU pragma: export
#include "./topology/async/neighborhoods.hpp"              // IWYU pragma: export
#include "./topology/async/orient_faces_consistently.hpp"  // IWYU pragma: export
#include "./topology/async/vertex_link.hpp"                // IWYU pragma: export
#include "./topology/boundary_curves.hpp"                  // IWYU pragma: export
#include "./topology/boundary_edges.hpp"                   // IWYU pragma: export
#include "./topology/boundary_paths.hpp"                   // IWYU pragma: export
#include "./topology/cdt.hpp"                              // IWYU pragma: export
#include "./topology/cell_membership.hpp"                  // IWYU pragma: export
#include "./topology/connect_edges_to_paths.hpp"           // IWYU pragma: export
#include "./topology/domain_labels.hpp"                    // IWYU pragma: export
#include "./topology/euler_characteristic.hpp"             // IWYU pragma: export
#include "./topology/face_link.hpp"                        // IWYU pragma: export
#include "./topology/is_closed.hpp"                        // IWYU pragma: export
#include "./topology/is_manifold.hpp"                      // IWYU pragma: export
#include "./topology/k_rings.hpp"                          // IWYU pragma: export
#include "./topology/label_connected_components.hpp"       // IWYU pragma: export
#include "./topology/manifold_edge_link.hpp"               // IWYU pragma: export
#include "./topology/neighborhoods.hpp"                    // IWYU pragma: export
#include "./topology/non_manifold_edges.hpp"               // IWYU pragma: export
#include "./topology/orient_faces_consistently.hpp"        // IWYU pragma: export
#include "./topology/vertex_link.hpp"                      // IWYU pragma: export
