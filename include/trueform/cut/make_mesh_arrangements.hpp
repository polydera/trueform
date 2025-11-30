/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../intersect/intersections_between_polygons.hpp"
#include "./impl/make_mesh_arrangements.hpp"
#include "./tagged_cut_faces.hpp"
namespace tf {
template <typename Policy0, typename Policy1>
auto make_mesh_arrangements(const tf::polygons<Policy0> _polygons0,
                            const tf::polygons<Policy1> &_polygons1) {
  using Index =
      std::common_type_t<typename Policy0::index_type, typename Policy1::index_type>;
  tf::intersections_between_polygons<Index, double, 3> ibp;
  ibp.build(tf::make_form(_polygons0), tf::make_form(_polygons1));
  tf::tagged_cut_faces<Index> tcf;
  tcf.build(_polygons0, _polygons1, ibp);
  return tf::cut::make_mesh_arrangements<int>(_polygons0, _polygons1, ibp, tcf);
}
} // namespace tf
