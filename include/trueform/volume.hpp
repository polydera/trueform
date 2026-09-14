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

/** @defgroup volume Volume Module
 *  Dense scalar fields on a regular grid: signed distance generation, boolean
 *  CSG, slicing and isosurface extraction.
 */

#include "./volume/boolean_op.hpp"            // IWYU pragma: export
#include "./volume/isosurface_config.hpp"     // IWYU pragma: export
#include "./volume/make_boolean.hpp"          // IWYU pragma: export
#include "./volume/make_isocontours.hpp"      // IWYU pragma: export
#include "./volume/make_isosurface.hpp"       // IWYU pragma: export
#include "./volume/make_mesh_sdf.hpp"         // IWYU pragma: export
#include "./volume/make_resampled_volume.hpp" // IWYU pragma: export
#include "./volume/make_sphere_sdf.hpp"       // IWYU pragma: export
#include "./volume/make_volume_slice.hpp"     // IWYU pragma: export
#include "./volume/mesh_sdf_config.hpp"       // IWYU pragma: export
#include "./volume/volume.hpp"                // IWYU pragma: export
#include "./volume/volume_buffer.hpp"         // IWYU pragma: export
