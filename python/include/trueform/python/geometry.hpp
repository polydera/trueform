/*
* Copyright (c) 2025 XLAB
* All rights reserved.
*
* This file is part of trueform (www.trueform.polydera.com)
*
* Licensed for noncommercial use under the PolyForm Noncommercial
* License 1.0.0.
* Commercial licensing available via info@polydera.com.
*
* Author: Žiga Sajovic
*/
#pragma once

#include <nanobind/nanobind.h>

namespace tf::py {

void register_fit_rigid_alignment(nanobind::module_ &m);

void register_fit_obb_alignment(nanobind::module_ &m);

void register_fit_knn_alignment(nanobind::module_ &m);

void register_chamfer_error(nanobind::module_ &m);

void register_triangulated(nanobind::module_ &m);

void register_geometry_module(nanobind::module_ &m);

} // namespace tf::py
