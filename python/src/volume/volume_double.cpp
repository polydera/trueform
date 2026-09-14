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
#include <trueform/python/io/nifti.hpp>
#include <trueform/python/volume/volume_ops_impl.hpp>

namespace tf::py {

auto register_volume_double(nanobind::module_ &m) -> void {
  register_volume_ops<double, double>(m, "VolumeWrapperDouble", "double");
  register_volume_boolean<double, double>(m, "double");
  register_volume_resample<double, double>(m, "double");
  register_volume_generators<double>(m, "double");
  register_volume_nifti<double, double>(m, "double");
}

} // namespace tf::py
