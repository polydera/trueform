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

#include <nanobind/ndarray.h>
#include <optional>
#include <trueform/core/form.hpp>
#include <trueform/core/frame.hpp>
#include <trueform/core/policy/frame.hpp>
#include <trueform/core/policy/normals.hpp>
#include <trueform/core/unit_vectors.hpp>
#include <trueform/geometry/fit_knn_alignment.hpp>
#include <trueform/python/spatial/point_cloud.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/spatial/policy/tree.hpp>

namespace tf::py {

// Point-to-point alignment
template <typename RealT, std::size_t Dims>
auto fit_knn_alignment_impl(point_cloud_wrapper<RealT, Dims> &cloud0,
                            point_cloud_wrapper<RealT, Dims> &cloud1,
                            std::size_t k, std::optional<float> sigma_opt,
                            float outlier_proportion) {
  float sigma = sigma_opt.value_or(-1.f);
  auto pts0 = cloud0.make_primitive_range();
  auto pts1 = cloud1.make_primitive_range();

  bool has0 = cloud0.has_transformation();
  bool has1 = cloud1.has_transformation();

  // tree() auto-builds if needed
  auto form1 = pts1 | tf::tag(cloud1.tree());

  tf::knn_alignment_config config{k, sigma, outlier_proportion};

  auto compute = [&]() {
    if (has0 && has1) {
      return tf::fit_knn_alignment(
          pts0 | tf::tag(tf::make_frame(cloud0.transformation_view())),
          form1 | tf::tag(tf::make_frame(cloud1.transformation_view())),
          config);
    } else if (has0) {
      return tf::fit_knn_alignment(
          pts0 | tf::tag(tf::make_frame(cloud0.transformation_view())), form1,
          config);
    } else if (has1) {
      return tf::fit_knn_alignment(
          pts0, form1 | tf::tag(tf::make_frame(cloud1.transformation_view())),
          config);
    } else {
      return tf::fit_knn_alignment(pts0, form1, config);
    }
  };

  return make_numpy_array(compute());
}

// Point-to-plane alignment (target has normals) - 3D only
template <typename RealT>
auto fit_knn_alignment_p2plane_impl(
    point_cloud_wrapper<RealT, 3> &cloud0,
    point_cloud_wrapper<RealT, 3> &cloud1,
    nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, 3>>
        normals1,
    std::size_t k, std::optional<float> sigma_opt, float outlier_proportion) {
  float sigma = sigma_opt.value_or(-1.f);
  auto pts0 = cloud0.make_primitive_range();
  auto pts1 = cloud1.make_primitive_range();
  auto n1 = tf::make_unit_vectors<3>(
      tf::make_range(static_cast<RealT *>(normals1.data()),
                     normals1.shape(0) * 3));

  bool has0 = cloud0.has_transformation();
  bool has1 = cloud1.has_transformation();

  auto form1 = pts1 | tf::tag(cloud1.tree()) | tf::tag_normals(n1);

  tf::knn_alignment_config config{k, sigma, outlier_proportion};

  auto compute = [&]() {
    if (has0 && has1) {
      return tf::fit_knn_alignment(
          pts0 | tf::tag(tf::make_frame(cloud0.transformation_view())),
          form1 | tf::tag(tf::make_frame(cloud1.transformation_view())),
          config);
    } else if (has0) {
      return tf::fit_knn_alignment(
          pts0 | tf::tag(tf::make_frame(cloud0.transformation_view())), form1,
          config);
    } else if (has1) {
      return tf::fit_knn_alignment(
          pts0, form1 | tf::tag(tf::make_frame(cloud1.transformation_view())),
          config);
    } else {
      return tf::fit_knn_alignment(pts0, form1, config);
    }
  };

  return make_numpy_array(compute());
}

// Point-to-plane with normal weighting (both have normals) - 3D only
template <typename RealT>
auto fit_knn_alignment_weighted_impl(
    point_cloud_wrapper<RealT, 3> &cloud0,
    nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, 3>>
        normals0,
    point_cloud_wrapper<RealT, 3> &cloud1,
    nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, 3>>
        normals1,
    std::size_t k, std::optional<float> sigma_opt, float outlier_proportion) {
  float sigma = sigma_opt.value_or(-1.f);
  auto pts0 = cloud0.make_primitive_range();
  auto pts1 = cloud1.make_primitive_range();
  auto n0 = tf::make_unit_vectors<3>(
      tf::make_range(static_cast<RealT *>(normals0.data()),
                     normals0.shape(0) * 3));
  auto n1 = tf::make_unit_vectors<3>(
      tf::make_range(static_cast<RealT *>(normals1.data()),
                     normals1.shape(0) * 3));

  bool has0 = cloud0.has_transformation();
  bool has1 = cloud1.has_transformation();

  auto form1 = pts1 | tf::tag(cloud1.tree()) | tf::tag_normals(n1);

  tf::knn_alignment_config config{k, sigma, outlier_proportion};

  auto compute = [&]() {
    if (has0 && has1) {
      return tf::fit_knn_alignment(
          pts0 | tf::tag(tf::make_frame(cloud0.transformation_view())) |
              tf::tag_normals(n0),
          form1 | tf::tag(tf::make_frame(cloud1.transformation_view())),
          config);
    } else if (has0) {
      return tf::fit_knn_alignment(
          pts0 | tf::tag(tf::make_frame(cloud0.transformation_view())) |
              tf::tag_normals(n0),
          form1, config);
    } else if (has1) {
      return tf::fit_knn_alignment(
          pts0 | tf::tag_normals(n0),
          form1 | tf::tag(tf::make_frame(cloud1.transformation_view())),
          config);
    } else {
      return tf::fit_knn_alignment(pts0 | tf::tag_normals(n0), form1, config);
    }
  };

  return make_numpy_array(compute());
}

} // namespace tf::py
