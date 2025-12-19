/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./topology/boundary_edges.hpp"             // IWYU pragma: export
#include "./topology/boundary_paths.hpp"             // IWYU pragma: export
#include "./topology/components/finder.hpp"          // IWYU pragma: export
#include "./topology/connect_edges_to_paths.hpp"     // IWYU pragma: export
#include "./topology/connected_component_labels.hpp" // IWYU pragma: export
#include "./topology/connectivity_type.hpp"          // IWYU pragma: export
#include "./topology/directed_edge_id_in_face.hpp"   // IWYU pragma: export
#include "./topology/directed_edge_link.hpp"         // IWYU pragma: export
#include "./topology/edge_id_in_face.hpp"            // IWYU pragma: export
#include "./topology/edge_orientation.hpp"           // IWYU pragma: export
#include "./topology/face_edge_neighbors.hpp"        // IWYU pragma: export
#include "./topology/face_hole_relations.hpp"        // IWYU pragma: export
#include "./topology/face_link.hpp"                  // IWYU pragma: export
#include "./topology/face_membership.hpp"            // IWYU pragma: export
#include "./topology/find_eulerian_paths.hpp"        // IWYU pragma: export
#include "./topology/hole_patcher.hpp"               // IWYU pragma: export
#include "./topology/is_closed.hpp"                  // IWYU pragma: export
#include "./topology/label_connected_components.hpp" // IWYU pragma: export
#include "./topology/make_applier.hpp"               // IWYU pragma: export
#include "./topology/make_edge_connected_component_labels.hpp" // IWYU pragma: export
#include "./topology/make_manifold_edge_connected_component_labels.hpp" // IWYU pragma: export
#include "./topology/make_vertex_connected_component_labels.hpp" // IWYU pragma: export
#include "./topology/manifold_edge_link.hpp"        // IWYU pragma: export
#include "./topology/manifold_edge_peer.hpp"        // IWYU pragma: export
#include "./topology/non_manifold_edges.hpp"        // IWYU pragma: export
#include "./topology/non_simple_edges.hpp"          // IWYU pragma: export
#include "./topology/orient_faces_consistently.hpp" // IWYU pragma: export
#include "./topology/path_connector.hpp"            // IWYU pragma: export
#include "./topology/planar_embedding.hpp"          // IWYU pragma: export
#include "./topology/planar_graph_regions.hpp"      // IWYU pragma: export
#include "./topology/policy.hpp"                    // IWYU pragma: export
#include "./topology/reverse_winding.hpp"           // IWYU pragma: export
#include "./topology/scoped_face_membership.hpp"    // IWYU pragma: export
#include "./topology/scoped_id.hpp"                 // IWYU pragma: export
#include "./topology/set_component_labels.hpp"      // IWYU pragma: export
#include "./topology/set_type.hpp"                  // IWYU pragma: export
#include "./topology/stitched_face_membership.hpp"  // IWYU pragma: export
#include "./topology/stitched_manifold_edge_link.hpp" // IWYU pragma: export
#include "./topology/vertex_id_in_face.hpp"         // IWYU pragma: export
#include "./topology/vertex_link.hpp"               // IWYU pragma: export
#include "./topology/vertex_link_like.hpp"          // IWYU pragma: export
