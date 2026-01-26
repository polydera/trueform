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

#include "trueform/python/core/closest_metric_point_pair.hpp"
#include "trueform/python/core/make_primitives.hpp"
#include <nanobind/ndarray.h>
#include <trueform/core/closest_metric_point_pair.hpp>

namespace tf::py {

namespace {

// Convert metric_point_pair result to Python tuple
template <typename RealT, std::size_t Dims, typename ResultT>
auto metric_point_pair_to_tuple(const ResultT &result) {
  auto [dist2, pt0, pt1] = result;

  // Create numpy arrays for the two points
  RealT *pt0_data = new RealT[Dims];
  auto pt0_array =
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<Dims>>(
          pt0_data, {Dims}, nanobind::capsule(pt0_data, [](void *p) noexcept {
            delete[] static_cast<RealT *>(p);
          }));

  RealT *pt1_data = new RealT[Dims];
  auto pt1_array =
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<Dims>>(
          pt1_data, {Dims}, nanobind::capsule(pt1_data, [](void *p) noexcept {
            delete[] static_cast<RealT *>(p);
          }));

  // Copy data
  for (std::size_t i = 0; i < Dims; ++i) {
    pt0_array.data()[i] = pt0[i];
    pt1_array.data()[i] = pt1[i];
  }

  return nanobind::make_tuple(dist2, pt0_array, pt1_array);
}

} // anonymous namespace

auto register_core_closest_metric_point_pair(nanobind::module_ &m) -> void {
  // Point to Point (float, 2D)
  m.def("closest_metric_point_pair_point_point_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt1_data) {
          auto pt0 = make_point_from_array<2, float>(pt0_data);
          auto pt1 = make_point_from_array<2, float>(pt1_data);
          return metric_point_pair_to_tuple<float, 2>(
              tf::closest_metric_point_pair(pt0, pt1));
        });

  // Point to Point (float, 3D)
  m.def("closest_metric_point_pair_point_point_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt1_data) {
          auto pt0 = make_point_from_array<3, float>(pt0_data);
          auto pt1 = make_point_from_array<3, float>(pt1_data);
          return metric_point_pair_to_tuple<float, 3>(
              tf::closest_metric_point_pair(pt0, pt1));
        });

  // Point to Point (double, 2D)
  m.def("closest_metric_point_pair_point_point_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               pt0_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               pt1_data) {
          auto pt0 = make_point_from_array<2, double>(pt0_data);
          auto pt1 = make_point_from_array<2, double>(pt1_data);
          return metric_point_pair_to_tuple<double, 2>(
              tf::closest_metric_point_pair(pt0, pt1));
        });

  // Point to Point (double, 3D)
  m.def("closest_metric_point_pair_point_point_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt0_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt1_data) {
          auto pt0 = make_point_from_array<3, double>(pt0_data);
          auto pt1 = make_point_from_array<3, double>(pt1_data);
          return metric_point_pair_to_tuple<double, 3>(
              tf::closest_metric_point_pair(pt0, pt1));
        });

  // Point to Segment (float, 3D)
  m.def(
      "closest_metric_point_pair_point_segment_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
             pt_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             seg_data) {
        auto pt = make_point_from_array<3, float>(pt_data);
        auto seg = make_segment_from_array<3, float>(seg_data);
        return metric_point_pair_to_tuple<float, 3>(
            tf::closest_metric_point_pair(pt, seg));
      });

  // Point to Polygon (float, 3D)
  m.def("closest_metric_point_pair_point_polygon_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return metric_point_pair_to_tuple<float, 3>(
              tf::closest_metric_point_pair(pt, poly));
        });

  // Point to Ray (float, 3D)
  m.def(
      "closest_metric_point_pair_point_ray_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
             pt_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray_data) {
        auto pt = make_point_from_array<3, float>(pt_data);
        auto ray = make_ray_from_array<3, float>(ray_data);
        return metric_point_pair_to_tuple<float, 3>(
            tf::closest_metric_point_pair(pt, ray));
      });

  // Point to Segment (float, 2D)
  m.def(
      "closest_metric_point_pair_point_segment_float2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
             pt_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             seg_data) {
        auto pt = make_point_from_array<2, float>(pt_data);
        auto seg = make_segment_from_array<2, float>(seg_data);
        return metric_point_pair_to_tuple<float, 2>(
            tf::closest_metric_point_pair(pt, seg));
      });

  // Point to Segment (double, 2D)
  m.def(
      "closest_metric_point_pair_point_segment_double2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
             pt_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             seg_data) {
        auto pt = make_point_from_array<2, double>(pt_data);
        auto seg = make_segment_from_array<2, double>(seg_data);
        return metric_point_pair_to_tuple<double, 2>(
            tf::closest_metric_point_pair(pt, seg));
      });

  // Point to Segment (double, 3D)
  m.def(
      "closest_metric_point_pair_point_segment_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
             pt_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             seg_data) {
        auto pt = make_point_from_array<3, double>(pt_data);
        auto seg = make_segment_from_array<3, double>(seg_data);
        return metric_point_pair_to_tuple<double, 3>(
            tf::closest_metric_point_pair(pt, seg));
      });

  // Point to Polygon (float, 2D)
  m.def("closest_metric_point_pair_point_polygon_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto pt = make_point_from_array<2, float>(pt_data);
          auto poly = make_polygon_from_array<2, float>(poly_data);
          return metric_point_pair_to_tuple<float, 2>(
              tf::closest_metric_point_pair(pt, poly));
        });

  // Point to Polygon (double, 2D)
  m.def("closest_metric_point_pair_point_polygon_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto pt = make_point_from_array<2, double>(pt_data);
          auto poly = make_polygon_from_array<2, double>(poly_data);
          return metric_point_pair_to_tuple<double, 2>(
              tf::closest_metric_point_pair(pt, poly));
        });

  // Point to Polygon (double, 3D)
  m.def("closest_metric_point_pair_point_polygon_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto pt = make_point_from_array<3, double>(pt_data);
          auto poly = make_polygon_from_array<3, double>(poly_data);
          return metric_point_pair_to_tuple<double, 3>(
              tf::closest_metric_point_pair(pt, poly));
        });

  // Segment to Segment (float, 2D)
  m.def(
      "closest_metric_point_pair_segment_segment_float2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             seg0_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             seg1_data) {
        auto seg0 = make_segment_from_array<2, float>(seg0_data);
        auto seg1 = make_segment_from_array<2, float>(seg1_data);
        return metric_point_pair_to_tuple<float, 2>(
            tf::closest_metric_point_pair(seg0, seg1));
      });

  // Segment to Segment (float, 3D)
  m.def(
      "closest_metric_point_pair_segment_segment_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             seg0_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             seg1_data) {
        auto seg0 = make_segment_from_array<3, float>(seg0_data);
        auto seg1 = make_segment_from_array<3, float>(seg1_data);
        return metric_point_pair_to_tuple<float, 3>(
            tf::closest_metric_point_pair(seg0, seg1));
      });

  // Segment to Segment (double, 2D)
  m.def(
      "closest_metric_point_pair_segment_segment_double2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             seg0_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             seg1_data) {
        auto seg0 = make_segment_from_array<2, double>(seg0_data);
        auto seg1 = make_segment_from_array<2, double>(seg1_data);
        return metric_point_pair_to_tuple<double, 2>(
            tf::closest_metric_point_pair(seg0, seg1));
      });

  // Segment to Segment (double, 3D)
  m.def(
      "closest_metric_point_pair_segment_segment_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             seg0_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             seg1_data) {
        auto seg0 = make_segment_from_array<3, double>(seg0_data);
        auto seg1 = make_segment_from_array<3, double>(seg1_data);
        return metric_point_pair_to_tuple<double, 3>(
            tf::closest_metric_point_pair(seg0, seg1));
      });

  // Segment to Polygon (float, 2D)
  m.def(
      "closest_metric_point_pair_segment_polygon_float2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             seg_data,
         nanobind::ndarray<nanobind::numpy, const float> poly_data) {
        auto seg = make_segment_from_array<2, float>(seg_data);
        auto poly = make_polygon_from_array<2, float>(poly_data);
        return metric_point_pair_to_tuple<float, 2>(
            tf::closest_metric_point_pair(seg, poly));
      });

  // Segment to Polygon (float, 3D)
  m.def(
      "closest_metric_point_pair_segment_polygon_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             seg_data,
         nanobind::ndarray<nanobind::numpy, const float> poly_data) {
        auto seg = make_segment_from_array<3, float>(seg_data);
        auto poly = make_polygon_from_array<3, float>(poly_data);
        return metric_point_pair_to_tuple<float, 3>(
            tf::closest_metric_point_pair(seg, poly));
      });

  // Segment to Polygon (double, 2D)
  m.def(
      "closest_metric_point_pair_segment_polygon_double2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             seg_data,
         nanobind::ndarray<nanobind::numpy, const double> poly_data) {
        auto seg = make_segment_from_array<2, double>(seg_data);
        auto poly = make_polygon_from_array<2, double>(poly_data);
        return metric_point_pair_to_tuple<double, 2>(
            tf::closest_metric_point_pair(seg, poly));
      });

  // Segment to Polygon (double, 3D)
  m.def(
      "closest_metric_point_pair_segment_polygon_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             seg_data,
         nanobind::ndarray<nanobind::numpy, const double> poly_data) {
        auto seg = make_segment_from_array<3, double>(seg_data);
        auto poly = make_polygon_from_array<3, double>(poly_data);
        return metric_point_pair_to_tuple<double, 3>(
            tf::closest_metric_point_pair(seg, poly));
      });

  // Polygon to Polygon (float, 2D)
  m.def("closest_metric_point_pair_polygon_polygon_float2d",
        [](nanobind::ndarray<nanobind::numpy, const float> poly0_data,
           nanobind::ndarray<nanobind::numpy, const float> poly1_data) {
          auto poly0 = make_polygon_from_array<2, float>(poly0_data);
          auto poly1 = make_polygon_from_array<2, float>(poly1_data);
          return metric_point_pair_to_tuple<float, 2>(
              tf::closest_metric_point_pair(poly0, poly1));
        });

  // Polygon to Polygon (float, 3D)
  m.def("closest_metric_point_pair_polygon_polygon_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float> poly0_data,
           nanobind::ndarray<nanobind::numpy, const float> poly1_data) {
          auto poly0 = make_polygon_from_array<3, float>(poly0_data);
          auto poly1 = make_polygon_from_array<3, float>(poly1_data);
          return metric_point_pair_to_tuple<float, 3>(
              tf::closest_metric_point_pair(poly0, poly1));
        });

  // Polygon to Polygon (double, 2D)
  m.def("closest_metric_point_pair_polygon_polygon_double2d",
        [](nanobind::ndarray<nanobind::numpy, const double> poly0_data,
           nanobind::ndarray<nanobind::numpy, const double> poly1_data) {
          auto poly0 = make_polygon_from_array<2, double>(poly0_data);
          auto poly1 = make_polygon_from_array<2, double>(poly1_data);
          return metric_point_pair_to_tuple<double, 2>(
              tf::closest_metric_point_pair(poly0, poly1));
        });

  // Polygon to Polygon (double, 3D)
  m.def("closest_metric_point_pair_polygon_polygon_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double> poly0_data,
           nanobind::ndarray<nanobind::numpy, const double> poly1_data) {
          auto poly0 = make_polygon_from_array<3, double>(poly0_data);
          auto poly1 = make_polygon_from_array<3, double>(poly1_data);
          return metric_point_pair_to_tuple<double, 3>(
              tf::closest_metric_point_pair(poly0, poly1));
        });

  // ==== Point to Ray ====
  // Point to Ray (float, 2D)
  m.def(
      "closest_metric_point_pair_point_ray_float2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
             pt_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             ray_data) {
        auto pt = make_point_from_array<2, float>(pt_data);
        auto ray = make_ray_from_array<2, float>(ray_data);
        return metric_point_pair_to_tuple<float, 2>(
            tf::closest_metric_point_pair(pt, ray));
      });

  // Point to Ray (float, 3D)
  m.def(
      "closest_metric_point_pair_point_ray_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
             pt_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray_data) {
        auto pt = make_point_from_array<3, float>(pt_data);
        auto ray = make_ray_from_array<3, float>(ray_data);
        return metric_point_pair_to_tuple<float, 3>(
            tf::closest_metric_point_pair(pt, ray));
      });

  // Point to Ray (double, 2D)
  m.def(
      "closest_metric_point_pair_point_ray_double2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
             pt_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             ray_data) {
        auto pt = make_point_from_array<2, double>(pt_data);
        auto ray = make_ray_from_array<2, double>(ray_data);
        return metric_point_pair_to_tuple<double, 2>(
            tf::closest_metric_point_pair(pt, ray));
      });

  // Point to Ray (double, 3D)
  m.def(
      "closest_metric_point_pair_point_ray_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
             pt_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray_data) {
        auto pt = make_point_from_array<3, double>(pt_data);
        auto ray = make_ray_from_array<3, double>(ray_data);
        return metric_point_pair_to_tuple<double, 3>(
            tf::closest_metric_point_pair(pt, ray));
      });

  // ==== Point to Line ====
  // Point to Line (float, 2D)
  m.def(
      "closest_metric_point_pair_point_line_float2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2>>
             pt_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             line_data) {
        auto pt = make_point_from_array<2, float>(pt_data);
        auto line = make_line_from_array<2, float>(line_data);
        return metric_point_pair_to_tuple<float, 2>(
            tf::closest_metric_point_pair(pt, line));
      });

  // Point to Line (float, 3D)
  m.def(
      "closest_metric_point_pair_point_line_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
             pt_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             line_data) {
        auto pt = make_point_from_array<3, float>(pt_data);
        auto line = make_line_from_array<3, float>(line_data);
        return metric_point_pair_to_tuple<float, 3>(
            tf::closest_metric_point_pair(pt, line));
      });

  // Point to Line (double, 2D)
  m.def(
      "closest_metric_point_pair_point_line_double2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2>>
             pt_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             line_data) {
        auto pt = make_point_from_array<2, double>(pt_data);
        auto line = make_line_from_array<2, double>(line_data);
        return metric_point_pair_to_tuple<double, 2>(
            tf::closest_metric_point_pair(pt, line));
      });

  // Point to Line (double, 3D)
  m.def(
      "closest_metric_point_pair_point_line_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
             pt_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             line_data) {
        auto pt = make_point_from_array<3, double>(pt_data);
        auto line = make_line_from_array<3, double>(line_data);
        return metric_point_pair_to_tuple<double, 3>(
            tf::closest_metric_point_pair(pt, line));
      });

  // ==== Segment to Ray ====
  // Segment to Ray (float, 2D)
  m.def(
      "closest_metric_point_pair_segment_ray_float2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             seg_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             ray_data) {
        auto seg = make_segment_from_array<2, float>(seg_data);
        auto ray = make_ray_from_array<2, float>(ray_data);
        return metric_point_pair_to_tuple<float, 2>(
            tf::closest_metric_point_pair(seg, ray));
      });

  // Segment to Ray (float, 3D)
  m.def(
      "closest_metric_point_pair_segment_ray_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             seg_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray_data) {
        auto seg = make_segment_from_array<3, float>(seg_data);
        auto ray = make_ray_from_array<3, float>(ray_data);
        return metric_point_pair_to_tuple<float, 3>(
            tf::closest_metric_point_pair(seg, ray));
      });

  // Segment to Ray (double, 2D)
  m.def(
      "closest_metric_point_pair_segment_ray_double2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             seg_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             ray_data) {
        auto seg = make_segment_from_array<2, double>(seg_data);
        auto ray = make_ray_from_array<2, double>(ray_data);
        return metric_point_pair_to_tuple<double, 2>(
            tf::closest_metric_point_pair(seg, ray));
      });

  // Segment to Ray (double, 3D)
  m.def(
      "closest_metric_point_pair_segment_ray_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             seg_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray_data) {
        auto seg = make_segment_from_array<3, double>(seg_data);
        auto ray = make_ray_from_array<3, double>(ray_data);
        return metric_point_pair_to_tuple<double, 3>(
            tf::closest_metric_point_pair(seg, ray));
      });

  // ==== Segment to Line ====
  // Segment to Line (float, 2D)
  m.def(
      "closest_metric_point_pair_segment_line_float2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             seg_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             line_data) {
        auto seg = make_segment_from_array<2, float>(seg_data);
        auto line = make_line_from_array<2, float>(line_data);
        return metric_point_pair_to_tuple<float, 2>(
            tf::closest_metric_point_pair(seg, line));
      });

  // Segment to Line (float, 3D)
  m.def(
      "closest_metric_point_pair_segment_line_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             seg_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             line_data) {
        auto seg = make_segment_from_array<3, float>(seg_data);
        auto line = make_line_from_array<3, float>(line_data);
        return metric_point_pair_to_tuple<float, 3>(
            tf::closest_metric_point_pair(seg, line));
      });

  // Segment to Line (double, 2D)
  m.def(
      "closest_metric_point_pair_segment_line_double2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             seg_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             line_data) {
        auto seg = make_segment_from_array<2, double>(seg_data);
        auto line = make_line_from_array<2, double>(line_data);
        return metric_point_pair_to_tuple<double, 2>(
            tf::closest_metric_point_pair(seg, line));
      });

  // Segment to Line (double, 3D)
  m.def(
      "closest_metric_point_pair_segment_line_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             seg_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             line_data) {
        auto seg = make_segment_from_array<3, double>(seg_data);
        auto line = make_line_from_array<3, double>(line_data);
        return metric_point_pair_to_tuple<double, 3>(
            tf::closest_metric_point_pair(seg, line));
      });

  // ==== Polygon to Ray ====
  // Polygon to Ray (float, 2D)
  m.def(
      "closest_metric_point_pair_polygon_ray_float2d",
      [](nanobind::ndarray<nanobind::numpy, const float> poly_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             ray_data) {
        auto poly = make_polygon_from_array<2, float>(poly_data);
        auto ray = make_ray_from_array<2, float>(ray_data);
        return metric_point_pair_to_tuple<float, 2>(
            tf::closest_metric_point_pair(poly, ray));
      });

  // Polygon to Ray (float, 3D)
  m.def(
      "closest_metric_point_pair_polygon_ray_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float> poly_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray_data) {
        auto poly = make_polygon_from_array<3, float>(poly_data);
        auto ray = make_ray_from_array<3, float>(ray_data);
        return metric_point_pair_to_tuple<float, 3>(
            tf::closest_metric_point_pair(poly, ray));
      });

  // Polygon to Ray (double, 2D)
  m.def(
      "closest_metric_point_pair_polygon_ray_double2d",
      [](nanobind::ndarray<nanobind::numpy, const double> poly_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             ray_data) {
        auto poly = make_polygon_from_array<2, double>(poly_data);
        auto ray = make_ray_from_array<2, double>(ray_data);
        return metric_point_pair_to_tuple<double, 2>(
            tf::closest_metric_point_pair(poly, ray));
      });

  // Polygon to Ray (double, 3D)
  m.def(
      "closest_metric_point_pair_polygon_ray_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double> poly_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray_data) {
        auto poly = make_polygon_from_array<3, double>(poly_data);
        auto ray = make_ray_from_array<3, double>(ray_data);
        return metric_point_pair_to_tuple<double, 3>(
            tf::closest_metric_point_pair(poly, ray));
      });

  // ==== Polygon to Line ====
  // Polygon to Line (float, 2D)
  m.def(
      "closest_metric_point_pair_polygon_line_float2d",
      [](nanobind::ndarray<nanobind::numpy, const float> poly_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             line_data) {
        auto poly = make_polygon_from_array<2, float>(poly_data);
        auto line = make_line_from_array<2, float>(line_data);
        return metric_point_pair_to_tuple<float, 2>(
            tf::closest_metric_point_pair(poly, line));
      });

  // Polygon to Line (float, 3D)
  m.def(
      "closest_metric_point_pair_polygon_line_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float> poly_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             line_data) {
        auto poly = make_polygon_from_array<3, float>(poly_data);
        auto line = make_line_from_array<3, float>(line_data);
        return metric_point_pair_to_tuple<float, 3>(
            tf::closest_metric_point_pair(poly, line));
      });

  // Polygon to Line (double, 2D)
  m.def(
      "closest_metric_point_pair_polygon_line_double2d",
      [](nanobind::ndarray<nanobind::numpy, const double> poly_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             line_data) {
        auto poly = make_polygon_from_array<2, double>(poly_data);
        auto line = make_line_from_array<2, double>(line_data);
        return metric_point_pair_to_tuple<double, 2>(
            tf::closest_metric_point_pair(poly, line));
      });

  // Polygon to Line (double, 3D)
  m.def(
      "closest_metric_point_pair_polygon_line_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double> poly_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             line_data) {
        auto poly = make_polygon_from_array<3, double>(poly_data);
        auto line = make_line_from_array<3, double>(line_data);
        return metric_point_pair_to_tuple<double, 3>(
            tf::closest_metric_point_pair(poly, line));
      });

  // ==== Ray to Ray ====
  // Ray to Ray (float, 2D)
  m.def(
      "closest_metric_point_pair_ray_ray_float2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             ray0_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             ray1_data) {
        auto ray0 = make_ray_from_array<2, float>(ray0_data);
        auto ray1 = make_ray_from_array<2, float>(ray1_data);
        return metric_point_pair_to_tuple<float, 2>(
            tf::closest_metric_point_pair(ray0, ray1));
      });

  // Ray to Ray (float, 3D)
  m.def(
      "closest_metric_point_pair_ray_ray_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray0_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray1_data) {
        auto ray0 = make_ray_from_array<3, float>(ray0_data);
        auto ray1 = make_ray_from_array<3, float>(ray1_data);
        return metric_point_pair_to_tuple<float, 3>(
            tf::closest_metric_point_pair(ray0, ray1));
      });

  // Ray to Ray (double, 2D)
  m.def(
      "closest_metric_point_pair_ray_ray_double2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             ray0_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             ray1_data) {
        auto ray0 = make_ray_from_array<2, double>(ray0_data);
        auto ray1 = make_ray_from_array<2, double>(ray1_data);
        return metric_point_pair_to_tuple<double, 2>(
            tf::closest_metric_point_pair(ray0, ray1));
      });

  // Ray to Ray (double, 3D)
  m.def(
      "closest_metric_point_pair_ray_ray_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray0_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray1_data) {
        auto ray0 = make_ray_from_array<3, double>(ray0_data);
        auto ray1 = make_ray_from_array<3, double>(ray1_data);
        return metric_point_pair_to_tuple<double, 3>(
            tf::closest_metric_point_pair(ray0, ray1));
      });

  // ==== Line to Line ====
  // Line to Line (float, 2D)
  m.def(
      "closest_metric_point_pair_line_line_float2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             line0_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             line1_data) {
        auto line0 = make_line_from_array<2, float>(line0_data);
        auto line1 = make_line_from_array<2, float>(line1_data);
        return metric_point_pair_to_tuple<float, 2>(
            tf::closest_metric_point_pair(line0, line1));
      });

  // Line to Line (float, 3D)
  m.def(
      "closest_metric_point_pair_line_line_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             line0_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             line1_data) {
        auto line0 = make_line_from_array<3, float>(line0_data);
        auto line1 = make_line_from_array<3, float>(line1_data);
        return metric_point_pair_to_tuple<float, 3>(
            tf::closest_metric_point_pair(line0, line1));
      });

  // Line to Line (double, 2D)
  m.def(
      "closest_metric_point_pair_line_line_double2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             line0_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             line1_data) {
        auto line0 = make_line_from_array<2, double>(line0_data);
        auto line1 = make_line_from_array<2, double>(line1_data);
        return metric_point_pair_to_tuple<double, 2>(
            tf::closest_metric_point_pair(line0, line1));
      });

  // Line to Line (double, 3D)
  m.def(
      "closest_metric_point_pair_line_line_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             line0_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             line1_data) {
        auto line0 = make_line_from_array<3, double>(line0_data);
        auto line1 = make_line_from_array<3, double>(line1_data);
        return metric_point_pair_to_tuple<double, 3>(
            tf::closest_metric_point_pair(line0, line1));
      });

  // ==== Ray to Line ====
  // Ray to Line (float, 2D)
  m.def(
      "closest_metric_point_pair_ray_line_float2d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             ray_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 2>>
             line_data) {
        auto ray = make_ray_from_array<2, float>(ray_data);
        auto line = make_line_from_array<2, float>(line_data);
        return metric_point_pair_to_tuple<float, 2>(
            tf::closest_metric_point_pair(ray, line));
      });

  // Ray to Line (float, 3D)
  m.def(
      "closest_metric_point_pair_ray_line_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             line_data) {
        auto ray = make_ray_from_array<3, float>(ray_data);
        auto line = make_line_from_array<3, float>(line_data);
        return metric_point_pair_to_tuple<float, 3>(
            tf::closest_metric_point_pair(ray, line));
      });

  // Ray to Line (double, 2D)
  m.def(
      "closest_metric_point_pair_ray_line_double2d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             ray_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 2>>
             line_data) {
        auto ray = make_ray_from_array<2, double>(ray_data);
        auto line = make_line_from_array<2, double>(line_data);
        return metric_point_pair_to_tuple<double, 2>(
            tf::closest_metric_point_pair(ray, line));
      });

  // Ray to Line (double, 3D)
  m.def(
      "closest_metric_point_pair_ray_line_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             line_data) {
        auto ray = make_ray_from_array<3, double>(ray_data);
        auto line = make_line_from_array<3, double>(line_data);
        return metric_point_pair_to_tuple<double, 3>(
            tf::closest_metric_point_pair(ray, line));
      });

  // ==== Point to Plane (3D only) ====
  m.def("closest_metric_point_pair_point_plane_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto pt = make_point_from_array<3, float>(pt_data);
          auto plane = make_plane_from_array<3, float>(plane_data);
          return metric_point_pair_to_tuple<float, 3>(
              tf::closest_metric_point_pair(pt, plane));
        });

  m.def("closest_metric_point_pair_point_plane_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane_data) {
          auto pt = make_point_from_array<3, double>(pt_data);
          auto plane = make_plane_from_array<3, double>(plane_data);
          return metric_point_pair_to_tuple<double, 3>(
              tf::closest_metric_point_pair(pt, plane));
        });

  // ==== Plane to Point (3D only) ====
  m.def("closest_metric_point_pair_plane_point_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<3>>
               pt_data) {
          auto plane = make_plane_from_array<3, float>(plane_data);
          auto pt = make_point_from_array<3, float>(pt_data);
          return metric_point_pair_to_tuple<float, 3>(
              tf::closest_metric_point_pair(plane, pt));
        });

  m.def("closest_metric_point_pair_plane_point_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<3>>
               pt_data) {
          auto plane = make_plane_from_array<3, double>(plane_data);
          auto pt = make_point_from_array<3, double>(pt_data);
          return metric_point_pair_to_tuple<double, 3>(
              tf::closest_metric_point_pair(plane, pt));
        });

  // ==== Segment to Plane (3D only) ====
  m.def(
      "closest_metric_point_pair_segment_plane_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             seg_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
             plane_data) {
        auto seg = make_segment_from_array<3, float>(seg_data);
        auto plane = make_plane_from_array<3, float>(plane_data);
        return metric_point_pair_to_tuple<float, 3>(
            tf::closest_metric_point_pair(seg, plane));
      });

  m.def(
      "closest_metric_point_pair_segment_plane_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             seg_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
             plane_data) {
        auto seg = make_segment_from_array<3, double>(seg_data);
        auto plane = make_plane_from_array<3, double>(plane_data);
        return metric_point_pair_to_tuple<double, 3>(
            tf::closest_metric_point_pair(seg, plane));
      });

  // ==== Plane to Segment (3D only) ====
  m.def(
      "closest_metric_point_pair_plane_segment_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
             plane_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             seg_data) {
        auto plane = make_plane_from_array<3, float>(plane_data);
        auto seg = make_segment_from_array<3, float>(seg_data);
        return metric_point_pair_to_tuple<float, 3>(
            tf::closest_metric_point_pair(plane, seg));
      });

  m.def(
      "closest_metric_point_pair_plane_segment_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
             plane_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             seg_data) {
        auto plane = make_plane_from_array<3, double>(plane_data);
        auto seg = make_segment_from_array<3, double>(seg_data);
        return metric_point_pair_to_tuple<double, 3>(
            tf::closest_metric_point_pair(plane, seg));
      });

  // ==== Ray to Plane (3D only) ====
  m.def(
      "closest_metric_point_pair_ray_plane_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
             plane_data) {
        auto ray = make_ray_from_array<3, float>(ray_data);
        auto plane = make_plane_from_array<3, float>(plane_data);
        return metric_point_pair_to_tuple<float, 3>(
            tf::closest_metric_point_pair(ray, plane));
      });

  m.def(
      "closest_metric_point_pair_ray_plane_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
             plane_data) {
        auto ray = make_ray_from_array<3, double>(ray_data);
        auto plane = make_plane_from_array<3, double>(plane_data);
        return metric_point_pair_to_tuple<double, 3>(
            tf::closest_metric_point_pair(ray, plane));
      });

  // ==== Plane to Ray (3D only) ====
  m.def(
      "closest_metric_point_pair_plane_ray_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
             plane_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             ray_data) {
        auto plane = make_plane_from_array<3, float>(plane_data);
        auto ray = make_ray_from_array<3, float>(ray_data);
        return metric_point_pair_to_tuple<float, 3>(
            tf::closest_metric_point_pair(plane, ray));
      });

  m.def(
      "closest_metric_point_pair_plane_ray_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
             plane_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             ray_data) {
        auto plane = make_plane_from_array<3, double>(plane_data);
        auto ray = make_ray_from_array<3, double>(ray_data);
        return metric_point_pair_to_tuple<double, 3>(
            tf::closest_metric_point_pair(plane, ray));
      });

  // ==== Line to Plane (3D only) ====
  m.def(
      "closest_metric_point_pair_line_plane_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             line_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
             plane_data) {
        auto line = make_line_from_array<3, float>(line_data);
        auto plane = make_plane_from_array<3, float>(plane_data);
        return metric_point_pair_to_tuple<float, 3>(
            tf::closest_metric_point_pair(line, plane));
      });

  m.def(
      "closest_metric_point_pair_line_plane_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             line_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
             plane_data) {
        auto line = make_line_from_array<3, double>(line_data);
        auto plane = make_plane_from_array<3, double>(plane_data);
        return metric_point_pair_to_tuple<double, 3>(
            tf::closest_metric_point_pair(line, plane));
      });

  // ==== Plane to Line (3D only) ====
  m.def(
      "closest_metric_point_pair_plane_line_float3d",
      [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
             plane_data,
         nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<2, 3>>
             line_data) {
        auto plane = make_plane_from_array<3, float>(plane_data);
        auto line = make_line_from_array<3, float>(line_data);
        return metric_point_pair_to_tuple<float, 3>(
            tf::closest_metric_point_pair(plane, line));
      });

  m.def(
      "closest_metric_point_pair_plane_line_double3d",
      [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
             plane_data,
         nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<2, 3>>
             line_data) {
        auto plane = make_plane_from_array<3, double>(plane_data);
        auto line = make_line_from_array<3, double>(line_data);
        return metric_point_pair_to_tuple<double, 3>(
            tf::closest_metric_point_pair(plane, line));
      });

  // ==== Polygon to Plane (3D only) ====
  m.def("closest_metric_point_pair_polygon_plane_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float> poly_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data) {
          auto poly = make_polygon_from_array<3, float>(poly_data);
          auto plane = make_plane_from_array<3, float>(plane_data);
          return metric_point_pair_to_tuple<float, 3>(
              tf::closest_metric_point_pair(poly, plane));
        });

  m.def("closest_metric_point_pair_polygon_plane_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double> poly_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane_data) {
          auto poly = make_polygon_from_array<3, double>(poly_data);
          auto plane = make_plane_from_array<3, double>(plane_data);
          return metric_point_pair_to_tuple<double, 3>(
              tf::closest_metric_point_pair(poly, plane));
        });

  // ==== Plane to Polygon (3D only) ====
  m.def("closest_metric_point_pair_plane_polygon_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane_data,
           nanobind::ndarray<nanobind::numpy, const float> poly_data) {
          auto plane = make_plane_from_array<3, float>(plane_data);
          auto poly = make_polygon_from_array<3, float>(poly_data);
          return metric_point_pair_to_tuple<float, 3>(
              tf::closest_metric_point_pair(plane, poly));
        });

  m.def("closest_metric_point_pair_plane_polygon_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane_data,
           nanobind::ndarray<nanobind::numpy, const double> poly_data) {
          auto plane = make_plane_from_array<3, double>(plane_data);
          auto poly = make_polygon_from_array<3, double>(poly_data);
          return metric_point_pair_to_tuple<double, 3>(
              tf::closest_metric_point_pair(plane, poly));
        });

  // ==== Plane to Plane (3D only) ====
  m.def("closest_metric_point_pair_plane_plane_float3d",
        [](nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane0_data,
           nanobind::ndarray<nanobind::numpy, const float, nanobind::shape<4>>
               plane1_data) {
          auto plane0 = make_plane_from_array<3, float>(plane0_data);
          auto plane1 = make_plane_from_array<3, float>(plane1_data);
          return metric_point_pair_to_tuple<float, 3>(
              tf::closest_metric_point_pair(plane0, plane1));
        });

  m.def("closest_metric_point_pair_plane_plane_double3d",
        [](nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane0_data,
           nanobind::ndarray<nanobind::numpy, const double, nanobind::shape<4>>
               plane1_data) {
          auto plane0 = make_plane_from_array<3, double>(plane0_data);
          auto plane1 = make_plane_from_array<3, double>(plane1_data);
          return metric_point_pair_to_tuple<double, 3>(
              tf::closest_metric_point_pair(plane0, plane1));
        });
}

} // namespace tf::py
