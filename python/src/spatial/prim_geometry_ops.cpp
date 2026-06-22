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
 * Author: Ziga Sajovic
 */

#include "trueform/python/core/prim_dispatch.hpp"
#include "trueform/python/util/make_numpy_array.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <trueform/core/algorithm/parallel_for_each.hpp>
#include <trueform/core/area.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/memory.hpp>
#include <trueform/core/normal.hpp>
#include <trueform/core/views/sequence_range.hpp>
#include <trueform/geometry/compute_normals.hpp>
#include <string>

namespace tf::py {

namespace {

template <std::size_t Dims, typename RealT>
auto register_prim_geometry_ops_impl(nanobind::module_ &m, const char *suffix)
    -> void {
  namespace nb = nanobind;
  using PW = primitive_wrapper<Dims, RealT>;

  auto name = [suffix](const char *base) -> std::string {
    return std::string(base) + suffix;
  };

  // ==== area ====
  m.def(
      name("area_prim_").c_str(),
      [](const PW &a) -> nb::object {
        auto type = a.type();
        if (type != prim_type::triangle && type != prim_type::polygon)
          throw nanobind::type_error(
              "area() only supports Triangle and Polygon primitives");

        if (!a.is_batch()) {
          RealT result;
          if (type == prim_type::triangle) {
            result = static_cast<RealT>(tf::area(make_triangle(a)));
          } else {
            result = static_cast<RealT>(tf::area(make_polygon(a)));
          }
          return nb::cast(result);
        }

        // Batch
        int n = a.count();
        tf::buffer<RealT> out;
        out.allocate(n);
        auto *dst = out.data();

        auto compute = [&](int i) {
          auto ptr = a.element_ptr(i);
          if (type == prim_type::triangle) {
            auto pts =
                tf::make_points<Dims>(tf::make_range(ptr, 3 * Dims));
            dst[i] = static_cast<RealT>(tf::area(tf::make_polygon<3>(pts)));
          } else {
            auto nv = a.poly_verts();
            auto pts = tf::make_points<Dims>(tf::make_range(
                ptr, static_cast<std::size_t>(nv) * Dims));
            dst[i] = static_cast<RealT>(tf::area(tf::make_polygon(pts)));
          }
        };

        if (n >= 5000)
          tf::parallel_for_each(tf::make_sequence_range(n), compute);
        else
          for (int i = 0; i < n; ++i)
            compute(i);

        return nb::cast(make_numpy_array<nb::shape<-1>>(
            std::move(out), {static_cast<size_t>(n)}));
      },
      nb::arg("a"));

  // ==== normals (3D only) ====
  if constexpr (Dims == 3) {
    m.def(
        name("normals_prim_").c_str(),
        [](const PW &a) -> nb::object {
          auto type = a.type();
          if (type != prim_type::triangle && type != prim_type::polygon)
            throw nanobind::type_error(
                "normals() only supports 3D Triangle and Polygon primitives");

          if (!a.is_batch()) {
            // Single: extract first 3 vertices for normal
            if (type == prim_type::triangle) {
              auto poly = make_triangle(a);
              auto n = tf::make_normal(poly[0], poly[1], poly[2]);
              auto *data = tf::allocate<RealT>(3);
              for (std::size_t j = 0; j < 3; ++j)
                data[j] = n[j];
              return nb::cast(make_numpy_array<nb::shape<3>>(data, {3}));
            } else {
              auto poly = make_polygon(a);
              auto n = tf::make_normal(poly[0], poly[1], poly[2]);
              auto *data = tf::allocate<RealT>(3);
              for (std::size_t j = 0; j < 3; ++j)
                data[j] = n[j];
              return nb::cast(make_numpy_array<nb::shape<3>>(data, {3}));
            }
          }

          // Batch: use compute_normals on polygons range
          if (type == prim_type::triangle) {
            auto polys = make_triangles(a);
            auto normals_buf = tf::compute_normals(polys);
            return nb::cast(make_numpy_array(std::move(normals_buf)));
          } else {
            auto polys = make_polygons(a);
            auto normals_buf = tf::compute_normals(polys);
            return nb::cast(make_numpy_array(std::move(normals_buf)));
          }
        },
        nb::arg("a"));
  }
}

} // anonymous namespace

auto register_prim_geometry_ops(nanobind::module_ &m) -> void {
  register_prim_geometry_ops_impl<2, float>(m, "float2d");
  register_prim_geometry_ops_impl<3, float>(m, "float3d");
  register_prim_geometry_ops_impl<2, double>(m, "double2d");
  register_prim_geometry_ops_impl<3, double>(m, "double3d");
}

} // namespace tf::py
