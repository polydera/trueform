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

auto register_volume_float(nanobind::module_ &m) -> void {
  register_volume_ops<float, float>(m, "VolumeWrapperFloat", "float");
  register_volume_boolean<float, float>(m, "float");
  register_volume_resample<float, float>(m, "float");
  register_volume_generators<float>(m, "float");
  register_volume_nifti<float, float>(m, "float");
}

} // namespace tf::py
