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

/// @file
/// @ingroup topology_policies
/// @brief Policy tag helpers for topology connectivity structures.
///
/// Provides functions to attach topology metadata (face membership, vertex link,
/// edge membership, etc.) to range types using a policy-based composition pattern.
/// Use `tf::tag()` or specific `tf::tag_*()` functions to attach topology data.

#pragma once

#include "./policy/edge_membership.hpp"    // IWYU pragma: export
#include "./policy/face_link.hpp"          // IWYU pragma: export
#include "./policy/face_membership.hpp"    // IWYU pragma: export
#include "./policy/manifold_edge_link.hpp" // IWYU pragma: export
#include "./policy/vertex_link.hpp"        // IWYU pragma: export
