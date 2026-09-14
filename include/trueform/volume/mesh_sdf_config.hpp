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

/// @ingroup volume
/// @brief How @ref tf::make_mesh_sdf computes the field's magnitude.
///
/// `exact` measures every sample against the surface: the default, and the
/// SDF an entry by that name promises. `banded` measures only the samples
/// within a band of the surface and propagates the far field with a seeded
/// distance sweep — an order of magnitude faster at routine grids; the
/// accuracy contract is @ref tf::make_mesh_sdf's. The sign is exact in
/// both.
enum class mesh_sdf_mode {
  exact,
  banded,
};

/// @ingroup volume
/// @brief Parameters of a mesh SDF sampling.
///
/// Implicitly constructible from a @ref tf::mesh_sdf_mode, so a call site
/// that only chooses the mode writes it directly.
struct mesh_sdf_config {
  mesh_sdf_config() = default;
  mesh_sdf_config(mesh_sdf_mode m) : mode(m) {}
  mesh_sdf_config(mesh_sdf_mode m, int band_) : mode(m), band(band_) {}

  /// @brief Which magnitude the field carries.
  mesh_sdf_mode mode = mesh_sdf_mode::exact;

  /// @brief Banded only: voxels of exactly measured magnitude on each side
  /// of the surface. Exact ignores it.
  int band = 2;
};

} // namespace tf
