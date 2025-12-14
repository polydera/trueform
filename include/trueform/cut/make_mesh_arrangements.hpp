/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../intersect/intersections_between_polygons.hpp"
#include "./impl/dispatch.hpp"
#include "./impl/make_mesh_arrangements.hpp"
#include "./tagged_cut_faces.hpp"

namespace tf {

template <typename Policy0, typename Policy1>
auto make_mesh_arrangements(const tf::polygons<Policy0> &_polygons0,
                            const tf::polygons<Policy1> &_polygons1) {
  return cut::impl::boolean_dispatch(
      _polygons0, _polygons1, [](const auto &p0, const auto &p1) {
        using Index = std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                                         typename std::decay_t<decltype(p1)>::index_type>;
        tf::intersections_between_polygons<Index, double, 3> ibp;
        ibp.build(tf::make_form(p0), tf::make_form(p1));
        tf::tagged_cut_faces<Index> tcf;
        tcf.build(p0, p1, ibp);
        return tf::cut::make_mesh_arrangements<int>(p0, p1, ibp, tcf);
      });
}

} // namespace tf
