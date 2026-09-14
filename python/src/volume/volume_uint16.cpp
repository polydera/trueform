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

auto register_volume_uint16(nanobind::module_ &m) -> void {
  register_volume_ops<std::uint16_t, float>(m, "VolumeWrapperUInt16", "uint16");
  register_volume_nifti<std::uint16_t, float>(m, "uint16");
}

} // namespace tf::py
