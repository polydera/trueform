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
#include <cstdint>
#include <trueform/python/io/nifti.hpp>
#include <trueform/python/volume/volume_ops_impl.hpp>

namespace tf::py {

auto register_volume_int16(nanobind::module_ &m) -> void {
  register_volume_ops<std::int16_t, float>(m, "VolumeWrapperInt16", "int16");
  register_volume_boolean<std::int16_t, float>(m, "int16");
  register_volume_nifti<std::int16_t, float>(m, "int16");
}

} // namespace tf::py
