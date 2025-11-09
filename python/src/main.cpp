/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#include <nanobind/nanobind.h>
#include "trueform/python/core.hpp"
#include "trueform/python/io.hpp"
#include "trueform/python/spatial.hpp"

namespace nb = nanobind;

NB_MODULE(_trueform, m) {
  m.doc() = "Python bindings for trueform geometric processing library";

  // Register all modules
  tf::py::register_core(m);
  tf::py::register_io(m);
  tf::py::register_spatial_module(m);
}
