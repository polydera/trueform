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

#pragma once

#include "trueform/python/core/prim_dispatch.hpp"
#include "trueform/python/util/make_numpy_array.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <trueform/core/buffer.hpp>
#include <trueform/core/frame.hpp>
#include <trueform/core/policy/frame.hpp>
#include <trueform/spatial/policy/tree.hpp>
#include <trueform/spatial/policy/winding.hpp>
#include <trueform/spatial/winding_number.hpp>
#include <cstddef>
#include <string>
#include <utility>

namespace tf::py {

// ============================================================================
// register_mesh_winding_number: the generalized winding number of a mesh at a
// point query. Mesh-only (needs the tree and its winding moments) and 3D-only.
// The number is dimensionless, not a coordinate, so it crosses as float64
// whatever the mesh's own real type.
//   single point → float, batch → ndarray[double]
// ============================================================================

template <typename FormWrapper, std::size_t Dims, typename RealT>
auto register_mesh_winding_number(nanobind::module_ &m, const char *suffix)
    -> void {
  namespace nb = nanobind;
  using PW = primitive_wrapper<Dims, RealT>;

  m.def(
      (std::string("winding_number_mesh_fp_") + suffix).c_str(),
      [](FormWrapper &fw, const PW &pw, double beta) -> nb::object {
        if (!pw.is_batch()) {
          auto run = [&](const auto &form) {
            return tf::winding_number(form, make_point(pw), {beta});
          };
          double w;
          if (fw.has_transformation())
            w = run(fw.make_primitive_range() | tf::tag(fw.tree()) |
                    tf::tag(fw.winding_moments()) |
                    tf::tag(tf::make_frame(fw.transformation_view())));
          else
            w = run(fw.make_primitive_range() | tf::tag(fw.tree()) |
                    tf::tag(fw.winding_moments()));
          return nb::cast(w);
        }

        auto n = static_cast<std::size_t>(pw.count());
        tf::buffer<double> out;
        out.allocate(n);

        auto run = [&](const auto &form) {
          tf::winding_number(form, make_points(pw), out, {beta});
        };

        if (fw.has_transformation())
          run(fw.make_primitive_range() | tf::tag(fw.tree()) |
              tf::tag(fw.winding_moments()) |
              tf::tag(tf::make_frame(fw.transformation_view())));
        else
          run(fw.make_primitive_range() | tf::tag(fw.tree()) |
              tf::tag(fw.winding_moments()));

        return nb::cast(
            make_numpy_array<nb::shape<-1>>(std::move(out), {n}));
      },
      nb::arg("form"), nb::arg("point"), nb::arg("beta") = 2.0);
}

} // namespace tf::py
