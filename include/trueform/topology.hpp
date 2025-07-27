/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./topology/boundary_edges.hpp"             // IWYU pragma: export
#include "./topology/boundary_paths.hpp"             // IWYU pragma: export
#include "./topology/components/finder.hpp"          // IWYU pragma: export
#include "./topology/connect_edges_to_paths.hpp"     // IWYU pragma: export
#include "./topology/directed_edge_id_in_face.hpp"   // IWYU pragma: export
#include "./topology/directed_edge_link.hpp"         // IWYU pragma: export
#include "./topology/edge_id_in_face.hpp"            // IWYU pragma: export
#include "./topology/edge_orientation.hpp"           // IWYU pragma: export
#include "./topology/face_edge_neighbors.hpp"        // IWYU pragma: export
#include "./topology/face_hole_relations.hpp"        // IWYU pragma: export
#include "./topology/face_link.hpp"                  // IWYU pragma: export
#include "./topology/face_membership.hpp"            // IWYU pragma: export
#include "./topology/find_eulerian_paths.hpp"        // IWYU pragma: export
#include "./topology/label_connected_components.hpp" // IWYU pragma: export
#include "./topology/make_applier.hpp"               // IWYU pragma: export
#include "./topology/manifold_edge_link.hpp"         // IWYU pragma: export
#include "./topology/manifold_edge_peer.hpp"         // IWYU pragma: export
#include "./topology/non_manifold_edges.hpp"         // IWYU pragma: export
#include "./topology/path_connector.hpp"             // IWYU pragma: export
#include "./topology/planar_embedding.hpp"           // IWYU pragma: export
#include "./topology/planar_graph_regions.hpp"       // IWYU pragma: export
#include "./topology/policy.hpp"                     // IWYU pragma: export
#include "./topology/scoped_face_membership.hpp"     // IWYU pragma: export
#include "./topology/scoped_id.hpp"                  // IWYU pragma: export
#include "./topology/vertex_link.hpp"                // IWYU pragma: export
