/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/aabb_from.hpp"
#include "../core/closest_metric_point.hpp"
#include "../core/closest_metric_point_pair.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/line_like.hpp"
#include "../core/polygon.hpp"
#include "../core/ray_like.hpp"
#include "../core/segment.hpp"
#include "./tree/traversal_metric.hpp"
#include "./tree_search/nearness_search.hpp"

namespace tf {

// ============================================================================
// Point overloads
// ============================================================================

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::point_like<Dims, Policy1> &obj) {
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      });
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::point_like<Dims, Policy1> &obj,
                     tf::coordinate_type<Policy0, Policy1> radius) {
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      radius);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::point_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &knn) {
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::point_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &&knn) {
  return neighbor_search(form, obj, knn);
}

// ============================================================================
// Segment overloads (use aabb_from for BV metric)
// ============================================================================

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::segment<Dims, Policy1> &obj) {
  auto obj_aabb = tf::aabb_from(obj);
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) {
        return tf::spatial::traversal_metric(bv, obj_aabb);
      },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      });
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::segment<Dims, Policy1> &obj,
                     tf::coordinate_type<Policy0, Policy1> radius) {
  auto obj_aabb = tf::aabb_from(obj);
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) {
        return tf::spatial::traversal_metric(bv, obj_aabb);
      },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      radius);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::segment<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &knn) {
  auto obj_aabb = tf::aabb_from(obj);
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) {
        return tf::spatial::traversal_metric(bv, obj_aabb);
      },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::segment<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &&knn) {
  return neighbor_search(form, obj, knn);
}

// ============================================================================
// Ray overloads
// ============================================================================

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::ray_like<Dims, Policy1> &obj) {
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      });
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::ray_like<Dims, Policy1> &obj,
                     tf::coordinate_type<Policy0, Policy1> radius) {
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      radius);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::ray_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &knn) {
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::ray_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &&knn) {
  return neighbor_search(form, obj, knn);
}

// ============================================================================
// Line overloads
// ============================================================================

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::line_like<Dims, Policy1> &obj) {
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      });
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::line_like<Dims, Policy1> &obj,
                     tf::coordinate_type<Policy0, Policy1> radius) {
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      radius);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::line_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &knn) {
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::line_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &&knn) {
  return neighbor_search(form, obj, knn);
}

// ============================================================================
// Plane overloads
// ============================================================================

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::plane_like<Dims, Policy1> &obj) {
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      });
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::plane_like<Dims, Policy1> &obj,
                     tf::coordinate_type<Policy0, Policy1> radius) {
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      radius);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::plane_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &knn) {
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) { return tf::spatial::traversal_metric(bv, obj); },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::plane_like<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &&knn) {
  return neighbor_search(form, obj, knn);
}

// ============================================================================
// Polygon overloads (3D uses tag_plane, 2D uses obj directly)
// ============================================================================

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::polygon<Dims, Policy1> &obj) {
  auto obj_aabb = tf::aabb_from(obj);
  auto plane_obj = tf::tag_plane(obj);
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) {
        return tf::spatial::traversal_metric(bv, obj_aabb);
      },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, plane_obj);
      });
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::polygon<Dims, Policy1> &obj,
                     tf::coordinate_type<Policy0, Policy1> radius) {
  auto obj_aabb = tf::aabb_from(obj);
  auto plane_obj = tf::tag_plane(obj);
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) {
        return tf::spatial::traversal_metric(bv, obj_aabb);
      },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, plane_obj);
      },
      radius);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::polygon<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &knn) {
  auto obj_aabb = tf::aabb_from(obj);
  auto plane_obj = tf::tag_plane(obj);
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) {
        return tf::spatial::traversal_metric(bv, obj_aabb);
      },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, plane_obj);
      },
      knn);
}

template <std::size_t Dims, typename Policy0, typename Policy1,
          typename RandomIt>
auto neighbor_search(const tf::form<Dims, Policy0> &form,
                     const tf::polygon<Dims, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &&knn) {
  return neighbor_search(form, obj, knn);
}

// 2D polygon overloads (use obj directly, not tag_plane)
template <typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<2, Policy0> &form,
                     const tf::polygon<2, Policy1> &obj) {
  auto obj_aabb = tf::aabb_from(obj);
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) {
        return tf::spatial::traversal_metric(bv, obj_aabb);
      },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      });
}

template <typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<2, Policy0> &form,
                     const tf::polygon<2, Policy1> &obj,
                     tf::coordinate_type<Policy0, Policy1> radius) {
  auto obj_aabb = tf::aabb_from(obj);
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) {
        return tf::spatial::traversal_metric(bv, obj_aabb);
      },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      radius);
}

template <typename Policy0, typename Policy1, typename RandomIt>
auto neighbor_search(const tf::form<2, Policy0> &form,
                     const tf::polygon<2, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &knn) {
  auto obj_aabb = tf::aabb_from(obj);
  return tf::spatial::nearness_search(
      form,
      [&](const auto &bv) {
        return tf::spatial::traversal_metric(bv, obj_aabb);
      },
      [&](const auto &primitive) {
        return tf::closest_metric_point(primitive, obj);
      },
      knn);
}

template <typename Policy0, typename Policy1, typename RandomIt>
auto neighbor_search(const tf::form<2, Policy0> &form,
                     const tf::polygon<2, Policy1> &obj,
                     tf::nearest_neighbors<RandomIt> &&knn) {
  return neighbor_search(form, obj, knn);
}

// ============================================================================
// Form-to-form overloads (dual tree)
// ============================================================================

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form0,
                     const tf::form<Dims, Policy1> &form1) {
  return tf::spatial::nearness_search(
      form0, form1, [](const auto &obj0, const auto &obj1) {
        return tf::closest_metric_point_pair(obj0, obj1);
      });
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto neighbor_search(const tf::form<Dims, Policy0> &form0,
                     const tf::form<Dims, Policy1> &form1,
                     tf::coordinate_type<Policy0, Policy1> radius) {
  return tf::spatial::nearness_search(
      form0, form1,
      [](const auto &obj0, const auto &obj1) {
        return tf::closest_metric_point_pair(obj0, obj1);
      },
      radius);
}

} // namespace tf
