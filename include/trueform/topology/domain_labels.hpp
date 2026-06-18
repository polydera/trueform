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
#include "../core/blocked_buffer.hpp"
namespace tf {

/// @ingroup topology_components
/// @brief Per-face per-side domain labels of a non-manifold polygon mesh.
///
/// A non-manifold surface mesh bounds multiple 3D regions ("domains").
/// Each face has two sides; each side bounds one domain.
///
/// Storage: `labels` is a @ref tf::blocked_buffer with block size 2,
/// one block per face. For face `f`:
///   - `labels[f][0]` = domain id of the bounded region that contains
///     `f` as a boundary face with REVERSED winding (equivalently, the
///     side that `f`'s stored normal points INTO).
///   - `labels[f][1]` = domain id of the bounded region that contains
///     `f` as a boundary face with FORWARD winding (equivalently, the
///     side that `f`'s stored normal points AWAY FROM).
///
/// @ref tf::split_into_domains uses this convention to emit watertight
/// outward-oriented submeshes: side-0 emissions reverse the stored
/// winding, side-1 emissions keep it.
///
/// @tparam LabelType The integer type for domain labels.
template <typename LabelType> struct domain_labels {
  /// @brief Per-face per-side domain labels. `labels[f][s]` is the
  /// domain id on side `s` (0 or 1) of face `f`.
  tf::blocked_buffer<LabelType, 2> labels;
  /// @brief Total number of distinct volumetric domains.
  LabelType n_domains;
  /// @brief The outer-shell (unbounded universe) domain. Under
  /// @ref tf::domain_config::exclude_outer_shell its sides are folded to
  /// the sentinel `n_domains` (one past the valid range), and this
  /// equals `n_domains`. Otherwise it is the index of the outer-shell
  /// domain in `[0, n_domains)`, or `-1` if there is no unbounded domain.
  LabelType outer_shell_label;
};
} // namespace tf
