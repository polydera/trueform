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

#include <nanobind/nanobind.h>
#include "trueform/python/clean.hpp"
#include "trueform/python/core.hpp"
#include "trueform/python/cut.hpp"
#include "trueform/python/geometry.hpp"
#include "trueform/python/intersect.hpp"
#include "trueform/python/io.hpp"
#include "trueform/python/reindex.hpp"
#include "trueform/python/spatial.hpp"
#include "trueform/python/topology.hpp"

namespace nb = nanobind;

NB_MODULE(_trueform, m) {
  m.doc() = "Python bindings for trueform geometric processing library";

  // Suppress nanobind leak warnings - these are false positives from VTK's
  // event loop holding references during Python shutdown
  nb::set_leak_warnings(false);

  // Register all modules
  tf::py::register_clean(m);
  tf::py::register_core(m);
  tf::py::register_cut(m);
  tf::py::register_geometry_module(m);
  tf::py::register_intersect(m);
  tf::py::register_io(m);
  tf::py::register_reindex(m);
  tf::py::register_spatial_module(m);
  tf::py::register_topology(m);
}
