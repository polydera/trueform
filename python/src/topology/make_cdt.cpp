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

#include "trueform/python/topology/make_cdt_impl.hpp"
#include <nanobind/nanobind.h>

namespace tf::py {

auto register_topology_make_cdt(nanobind::module_ &m) -> void {
  using namespace nanobind;

  m.def("make_cdt_float", &make_cdt_impl<float>, arg("points"));
  m.def("make_cdt_double", &make_cdt_impl<double>, arg("points"));

  m.def("make_cdt_float_with_maps", &make_cdt_with_maps_impl<float>,
        arg("points"));
  m.def("make_cdt_double_with_maps", &make_cdt_with_maps_impl<double>,
        arg("points"));

  m.def("make_cdt_float_edges", &make_cdt_edges_impl<float>, arg("points"),
        arg("edges"), arg("edge_mask").none() = nanobind::none(),
        arg("split_constraints") = true);
  m.def("make_cdt_double_edges", &make_cdt_edges_impl<double>, arg("points"),
        arg("edges"), arg("edge_mask").none() = nanobind::none(),
        arg("split_constraints") = true);

  m.def("make_cdt_float_edges_with_maps",
        &make_cdt_edges_with_maps_impl<float>, arg("points"), arg("edges"),
        arg("edge_mask").none() = nanobind::none(),
        arg("split_constraints") = true);
  m.def("make_cdt_double_edges_with_maps",
        &make_cdt_edges_with_maps_impl<double>, arg("points"), arg("edges"),
        arg("edge_mask").none() = nanobind::none(),
        arg("split_constraints") = true);

  m.def("make_cdt_float_edges_labels", &make_cdt_edges_labels_impl<float>,
        arg("points"), arg("edges"),
        arg("edge_mask").none() = nanobind::none(),
        arg("split_constraints") = true, arg("region_mode") = 0);
  m.def("make_cdt_double_edges_labels", &make_cdt_edges_labels_impl<double>,
        arg("points"), arg("edges"),
        arg("edge_mask").none() = nanobind::none(),
        arg("split_constraints") = true, arg("region_mode") = 0);

  m.def("make_cdt_float_edges_labels_with_maps",
        &make_cdt_edges_labels_with_maps_impl<float>, arg("points"),
        arg("edges"), arg("edge_mask").none() = nanobind::none(),
        arg("split_constraints") = true, arg("region_mode") = 0);
  m.def("make_cdt_double_edges_labels_with_maps",
        &make_cdt_edges_labels_with_maps_impl<double>, arg("points"),
        arg("edges"), arg("edge_mask").none() = nanobind::none(),
        arg("split_constraints") = true, arg("region_mode") = 0);
}

} // namespace tf::py
