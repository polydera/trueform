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
#include <trueform/core/algorithm/parallel_for_each.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/checked.hpp>
#include <trueform/core/frame.hpp>
#include <trueform/core/policy/frame.hpp>
#include <trueform/core/views/sequence_range.hpp>
#include <trueform/spatial/policy/tree.hpp>
#include <trueform/spatial/policy/winding.hpp>
#include <trueform/spatial/signed_distance.hpp>
#include <cstddef>
#include <string>

namespace tf::py {

// ============================================================================
// register_mesh_signed_distance: the winding-signed distance of a mesh to a
// point query. Mesh-only (needs the tree and its winding moments) and 3D-only;
// the sign is the generalized winding number, negative inside.
//   single point → scalar, batch → ndarray[RealT]
// ============================================================================

template <typename FormWrapper, std::size_t Dims, typename RealT>
auto register_mesh_signed_distance(nanobind::module_ &m, const char *suffix)
    -> void {
  namespace nb = nanobind;
  using PW = primitive_wrapper<Dims, RealT>;

  m.def(
      (std::string("signed_distance_mesh_fp_") + suffix).c_str(),
      [](FormWrapper &fw, const PW &pw) -> nb::object {
        if (!pw.is_batch()) {
          auto run = [&](const auto &form) -> RealT {
            return static_cast<RealT>(
                tf::signed_distance(form, make_point(pw)));
          };
          RealT d;
          if (fw.has_transformation())
            d = run(fw.make_primitive_range() | tf::tag(fw.tree()) |
                    tf::tag(fw.winding_moments()) |
                    tf::tag(tf::make_frame(fw.transformation_view())));
          else
            d = run(fw.make_primitive_range() | tf::tag(fw.tree()) |
                    tf::tag(fw.winding_moments()));
          return nb::cast(d);
        }

        fw.winding_moments();
        int n = pw.count();
        tf::buffer<RealT> out;
        out.allocate(n);
        auto *dst = out.data();

        auto run = [&](const auto &form) {
          auto compute = [&](int i) {
            auto pt = tf::make_point_view<Dims>(pw.element_ptr(i));
            dst[i] = static_cast<RealT>(tf::signed_distance(form, pt));
          };
          tf::parallel_for_each(tf::make_sequence_range(n), compute,
                                tf::checked);
        };

        if (fw.has_transformation())
          run(fw.make_primitive_range() | tf::tag(fw.tree()) |
              tf::tag(fw.winding_moments()) |
              tf::tag(tf::make_frame(fw.transformation_view())));
        else
          run(fw.make_primitive_range() | tf::tag(fw.tree()) |
              tf::tag(fw.winding_moments()));

        return nb::cast(make_numpy_array<nb::shape<-1>>(
            std::move(out), {static_cast<size_t>(n)}));
      },
      nb::arg("form"), nb::arg("point"));
}

} // namespace tf::py
