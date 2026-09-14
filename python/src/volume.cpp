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

#include "trueform/python/volume.hpp"

namespace tf::py {

auto register_volume(nanobind::module_ &m) -> void {
  // Create volume submodule
  auto volume_module =
      m.def_submodule("volume", "Scalar volume (regular voxel grid) operations");

  // Register volume components to submodule
  register_volume_float(volume_module);
  register_volume_double(volume_module);
  register_volume_int16(volume_module);
  register_volume_uint16(volume_module);
  register_volume_uint8(volume_module);
  register_volume_make_mesh_sdf(volume_module);
}

} // namespace tf::py
