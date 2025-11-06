/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../classify/tagged.hpp"
#include "./ids_common.hpp"
#include "./make_boolean_common.hpp"

namespace tf::cut {

template <typename LabelType, typename Policy0, typename Policy1,
          typename Index, typename RealT, std::size_t Dims>
auto make_mesh_arrangements(
    const tf::polygons<Policy0> _polygons0,
    const tf::polygons<Policy1> &_polygons1,
    const tf::intersect::tagged_intersections<Index, RealT, Dims> &ibp,
    const tf::tagged_cut_faces<Index> &tcf) {
  auto make_polygons = [](const auto &form) {
    return tf::wrap_map(form, [](auto &&x) {
      return tf::core::make_polygons(x.faces(),
                                     x.points().template as<RealT>());
    });
  };
  auto polygons0 = make_polygons(_polygons0);
  auto polygons1 = make_polygons(_polygons1);
  auto [pal0, pal1, counts0, counts1] =
      tf::cut::make_classification_counts<LabelType>(polygons0, polygons1, ibp,
                                                     tcf);
  tf::cut::polygon_arrangement_ids<Index> pai0;
  tf::cut::polygon_arrangement_ids<Index> pai1;
  tbb::parallel_invoke(
      [&pal0 = pal0, &pai0] {
        pai0 = tf::cut::make_polygon_arrangement_ids<Index>(pal0);
      },
      [&pal1 = pal1, &pai1] {
        pai1 = tf::cut::make_polygon_arrangement_ids<Index>(pal1);
      });

  using res_t = decltype(make_boolean_common(
      _polygons0, tf::make_points(ibp.intersection_points()), pai0,
      tcf.descriptors0(), tcf.mapped_loops0(), 0, tf::direction::forward));
  std::vector<res_t> out;
  out.resize(pai0.polygons.size() + pai1.polygons.size());

  tf::buffer<tf::strict_containment> containments;
  containments.allocate(out.size());
  tf::buffer<std::int8_t> labels;
  labels.allocate(out.size());

  tbb::task_group tg;

  for (std::size_t i = 0; i < std::size_t(pai0.polygons.size()); ++i) {
    tg.run([&, &counts0 = counts0, i] {
      out[i] = make_boolean_common(
          _polygons0, tf::make_points(ibp.intersection_points()), pai0,
          tcf.descriptors0(), tcf.mapped_loops0(), i, tf::direction::forward);
      containments[i] = counts0[i][0] < counts0[i][1]
                            ? tf::strict_containment::outside
                            : tf::strict_containment::inside;
      labels[i] = 0;
    });
  }

  for (std::size_t i = 0; i < std::size_t(pai1.polygons.size()); ++i) {
    tg.run([&, &counts1 = counts1, i,
            offs = std::size_t(pai0.polygons.size())] {
      out[i + offs] = make_boolean_common(
          _polygons1, tf::make_points(ibp.intersection_points()), pai1,
          tcf.descriptors1(), tcf.mapped_loops1(), i, tf::direction::forward);
      containments[i + offs] = counts1[i][0] < counts1[i][1]
                                   ? tf::strict_containment::outside
                                   : tf::strict_containment::inside;
      labels[i + offs] = 1;
    });
  }

  tg.wait();
  return std::make_tuple(std::move(out), std::move(labels),
                         std::move(containments));
}
} // namespace tf::cut
