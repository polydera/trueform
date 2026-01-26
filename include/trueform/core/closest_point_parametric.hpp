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
#include "./dot.hpp"
#include "./line.hpp"
#include "./line_line_check.hpp"
#include "./plane_like.hpp"
#include "./point_like.hpp"
#include "./ray_like.hpp"
#include "./segment.hpp"
#include <algorithm>

namespace tf {

/// @ingroup core_queries
/// @brief Computes the parametric location on a ray closest to a plane.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_point_parametric(const tf::ray_like<Dims, Policy0> &ray,
                              const tf::plane_like<Dims, Policy1> &plane) {
  auto Vd = tf::dot(plane.normal, ray.direction);
  auto V0 = tf::dot(plane.normal, ray.origin) + plane.d;
  decltype(Vd) t;
  if (Vd * Vd <
      ray.direction.length2() * std::numeric_limits<decltype(t)>::epsilon()) {
    t = 0;
  } else {
    t = -V0 / Vd;
    t *= (t > 0);
  }
  return t;
}

/// @ingroup core_queries
/// @brief Computes the parametric location on a line closest to a plane.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_point_parametric(const tf::line_like<Dims, Policy0> &line,
                              const tf::plane_like<Dims, Policy1> &plane) {
  auto Vd = tf::dot(plane.normal, line.direction);
  auto V0 = tf::dot(plane.normal, line.origin) + plane.d;
  decltype(Vd) t;
  if (Vd * Vd <
      line.direction.length2() * std::numeric_limits<decltype(t)>::epsilon()) {
    t = 0;
  } else {
    t = -V0 / Vd;
  }
  return t;
}

/// @ingroup core_queries
/// @brief Computes the parametric location on a ray closest to a plane.
template <std::size_t Dims, typename Policy, typename Policy1>
auto closest_point_parametric(const tf::segment<Dims, Policy> &segment,
                              const tf::plane_like<Dims, Policy1> &plane) {
  auto line = tf::make_line_between_points(segment[0], segment[1]);
  auto t = closest_point_parametric(line, plane);
  return std::clamp(t, decltype(t)(0), decltype(t)(1));
}

/// @ingroup core_queries
/// @brief Computes the parametric location on a segment closest to a point.
template <std::size_t Dims, typename Policy, typename T>
auto closest_point_parametric(const tf::ray_like<Dims, Policy> &ray,
                              const tf::point_like<Dims, T> &point) {
  auto dist_vec = point - ray.origin;
  auto t = tf::dot(dist_vec, ray.direction) / ray.direction.length2();
  return std::max(decltype(t)(0), t);
}

/// @ingroup core_queries
/// @brief Computes the parametric location on a line closest to a point.
template <std::size_t Dims, typename Policy, typename T>
auto closest_point_parametric(const tf::line_like<Dims, Policy> &line,
                              const tf::point_like<Dims, T> &point) {
  auto dist_vec = point - line.origin;
  auto t = tf::dot(dist_vec, line.direction) / line.direction.length2();
  return t;
}

/// @ingroup core_queries
/// @brief Computes the parametric location on a segment closest to a point.
template <std::size_t Dims, typename Policy0, typename T>
auto closest_point_parametric(const tf::segment<Dims, Policy0> &segment,
                              const tf::point_like<Dims, T> &point) {
  auto direction = segment[1] - segment[0];
  auto dist_vec = point - segment[0];
  auto t = tf::dot(dist_vec, direction) / direction.length2();
  return std::clamp(t, decltype(t)(0), decltype(t)(1));
}

/// @ingroup core_queries
/// @brief Computes the parametric locations of closest points on a ray, and a
/// ray.

template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_point_parametric(const tf::ray_like<Dims, Policy0> &ray0,
                              const tf::ray_like<Dims, Policy1> &ray1)
    -> std::pair<tf::coordinate_type<Policy0, Policy1>,
                 tf::coordinate_type<Policy0, Policy1>> {

  using RealT = tf::coordinate_type<Policy0, Policy1>;

  const auto &p0 = ray0.origin;
  const auto &d0 = ray0.direction;
  const auto &p1 = ray1.origin;
  const auto &d1 = ray1.direction;

  auto [status, t0_raw, t1_raw] = tf::core::line_line_check_full(ray0, ray1);

  RealT t0 = t0_raw;
  RealT t1 = t1_raw;

  switch (status) {
  case tf::intersect_status::non_parallel: {
    // t0, t1 = closest points on infinite lines.
    // Now clamp to rays (t >= 0) in a consistent way.

    if (t0 >= RealT(0) && t1 >= RealT(0)) {
      return {t0, t1};
    }

    // Both behind: closest are the origins.
    if (t0 < RealT(0) && t1 < RealT(0)) {
      return {RealT(0), RealT(0)};
    }

    // One behind, one valid: clamp the behind one to 0 and reproject the other.

    if (t0 < RealT(0)) {
      t0 = RealT(0);
      const auto w = p0 - p1;
      const auto d1_dot = tf::dot(d1, d1);
      if (d1_dot > RealT(0)) {
        t1 = tf::dot(w, d1) / d1_dot; // projection of p0 onto ray1
        if (t1 < RealT(0))
          t1 = RealT(0);
      } else {
        t1 = RealT(0);
      }
      return {t0, t1};
    }

    // t1 < 0
    t1 = RealT(0);
    {
      const auto w = p1 - p0;
      const auto d0_dot = tf::dot(d0, d0);
      if (d0_dot > RealT(0)) {
        t0 = tf::dot(w, d0) / d0_dot; // projection of p1 onto ray0
        if (t0 < RealT(0))
          t0 = RealT(0);
      } else {
        t0 = RealT(0);
      }
    }
    return {t0, t1};
  }

  case tf::intersect_status::parallel: {
    // Parallel, non-colinear.
    // Minimal segment is between one origin and the other ray.

    t0 = RealT(0);

    const auto w = p0 - p1;
    const auto d1_dot = tf::dot(d1, d1);

    if (d1_dot > RealT(0)) {
      t1 = tf::dot(w, d1) / d1_dot; // projection of p0 onto ray1
      if (t1 < RealT(0))
        t1 = RealT(0);
    } else {
      t1 = RealT(0);
    }

    return {t0, t1};
  }

  case tf::intersect_status::colinear: {
    // Colinear rays: closest distance is 0; choose a canonical pair.
    // Reduce to 1D along ray0.

    const auto d0_dot = tf::dot(d0, d0);
    if (d0_dot == RealT(0)) {
      // Degenerate ray0: fall back to both at origins.
      return {RealT(0), RealT(0)};
    }

    const RealT u1_on_0 = tf::dot(p1 - p0, d0) / d0_dot;

    if (u1_on_0 >= RealT(0)) {
      // ray1.origin lies "in front of" ray0 along d0:
      // take that shared point as closest: p1 vs ray0(u1_on_0).
      t0 = u1_on_0;
      t1 = RealT(0);
      return {t0, t1};
    }

    // ray1.origin is "behind" ray0 start: closest is at p0.
    // (either p0 vs p1, or they share origin; both give t0=t1=0)
    return {RealT(0), RealT(0)};
  }

  default:
    return {RealT(0), RealT(0)};
  }
}

/// @ingroup core_queries
/// @brief Computes the parametric locations of closest points on a ray, and a
/// line.

template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_point_parametric(const tf::ray_like<Dims, Policy0> &ray,
                              const tf::line_like<Dims, Policy1> &line)
    -> std::pair<tf::coordinate_type<Policy0, Policy1>,
                 tf::coordinate_type<Policy0, Policy1>> {

  using RealT = tf::coordinate_type<Policy0, Policy1>;

  const auto &p0 = ray.origin;
  const auto &p1 = line.origin;
  const auto &d1 = line.direction;

  auto [status, t0_raw, t1_raw] = tf::core::line_line_check_full(ray, line);

  RealT t0 = t0_raw;
  RealT t1 = t1_raw;

  switch (status) {
  case tf::intersect_status::non_parallel: {
    // t0,t1 are closest points on infinite lines.

    if (t0 >= RealT(0)) {
      // Already valid for the ray.
      return {t0, t1};
    }

    // Closest point on ray is at its origin (t0=0). Reproject that onto line.
    t0 = RealT(0);

    const auto w = p0 - p1;
    const auto d1_dot = tf::dot(d1, d1);
    if (d1_dot > RealT(0)) {
      t1 = tf::dot(w, d1) / d1_dot;
    } else {
      t1 = RealT(0); // degenerate line direction
    }
    return {t0, t1};
  }

  case tf::intersect_status::parallel: {
    // Parallel but not colinear: closest is from ray origin to line.
    t0 = RealT(0);

    const auto w = p0 - p1;
    const auto d1_dot = tf::dot(d1, d1);
    if (d1_dot > RealT(0)) {
      t1 = tf::dot(w, d1) / d1_dot; // projection of ray origin onto line
    } else {
      t1 = RealT(0);
    }
    return {t0, t1};
  }

  case tf::intersect_status::colinear: {
    // Colinear: distance is zero everywhere on the ray; choose a canonical
    // pair. Use the ray origin as the closest point on the ray.
    t0 = RealT(0);

    const auto d1_dot = tf::dot(d1, d1);
    if (d1_dot > RealT(0)) {
      // Param on line that hits the ray origin.
      t1 = tf::dot(p0 - p1, d1) / d1_dot;
    } else {
      t1 = RealT(0); // degenerate line, nothing better to do
    }
    return {t0, t1};
  }

  default:
    return {RealT(0), RealT(0)};
  }
}

/// @ingroup core_queries
/// @brief Computes the parametric locations of closest points on a line, and a
/// ray.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_point_parametric(const tf::line_like<Dims, Policy0> &line,
                              const tf::ray_like<Dims, Policy1> &ray) {
  auto out = closest_point_parametric(ray, line);
  std::swap(out.first, out.second);
  return out;
}

/// @ingroup core_queries
/// @brief Computes the parametric locations of closest points on a ray, and a
/// segment.
template <std::size_t Dims, typename Policy, typename T>
auto closest_point_parametric(const tf::ray_like<Dims, Policy> &ray,
                              const tf::segment<Dims, T> &segment)
    -> std::pair<tf::coordinate_type<Policy, T>,
                 tf::coordinate_type<Policy, T>> {
  using RealT = tf::coordinate_type<Policy, T>;

  const auto &ro = ray.origin;
  const auto &rd = ray.direction;

  auto line1 = tf::make_line_between_points(segment[0], segment[1]);
  const auto &s0 = segment[0];
  const auto  sd = segment[1] - segment[0];

  auto [status, t0_raw, t1_raw] = tf::core::line_line_check_full(ray, line1);

  const RealT rd2 = tf::dot(rd, rd);
  const RealT sd2 = tf::dot(sd, sd);

  auto clamp01 = [](RealT v) {
    return v < RealT(0) ? RealT(0) : (v > RealT(1) ? RealT(1) : v);
  };

  switch (status) {
  case tf::intersect_status::non_parallel: {
    RealT t0 = t0_raw;
    RealT t1 = t1_raw;

    // If closest points on infinite lines already lie on ray + segment, done.
    if (t0 >= RealT(0) && t1 >= RealT(0) && t1 <= RealT(1))
      return {t0, t1};

    // Ray side behind origin -> clamp ray to origin, project that to segment.
    if (t0 < RealT(0)) {
      t0 = RealT(0);
      if (sd2 > RealT(0)) {
        const auto proj = tf::dot(ro - s0, sd) / sd2;
        t1 = clamp01(proj);
      } else {
        t1 = RealT(0); // degenerate segment
      }
      return {t0, t1};
    }

    // Ray ok, segment param out of range -> clamp segment, reproject to ray.
    t1 = clamp01(t1);
    if (rd2 > RealT(0)) {
      const auto q = s0 + t1 * sd;
      t0 = tf::dot(q - ro, rd) / rd2;
      if (t0 < RealT(0))
        t0 = RealT(0);
    } else {
      t0 = RealT(0); // degenerate ray dir
    }
    return {t0, t1};
  }

  case tf::intersect_status::parallel: {
    // Parallel (non-colinear): closest is ray origin vs segment.
    RealT t0 = RealT(0);
    RealT t1 = RealT(0);
    if (sd2 > RealT(0)) {
      const auto proj = tf::dot(ro - s0, sd) / sd2;
      t1 = clamp01(proj);
    }
    return {t0, t1};
  }

  case tf::intersect_status::colinear: {
    // Colinear: both lie on same line. Handle with 1D params along rd.
    if (rd2 == RealT(0) || sd2 == RealT(0)) {
      // Degenerate ray or segment: nothing smarter to do.
      return {RealT(0), RealT(0)};
    }

    // Parameters of segment endpoints along the ray direction.
    const RealT a0 = tf::dot(s0 - ro, rd) / rd2;
    const RealT a1 = tf::dot(segment[1] - ro, rd) / rd2;
    const RealT lo = std::min(a0, a1);
    const RealT hi = std::max(a0, a1);

    if (hi < RealT(0)) {
      // Whole segment is behind ray origin.
      // Closest point: origin vs closer endpoint (larger param).
      RealT t0 = RealT(0);
      RealT t1 = (a0 > a1) ? RealT(0) : RealT(1);
      return {t0, t1};
    }

    // There is overlap or at least one endpoint at/after ray start.
    // Intersection of ray [0, +inf) with segment [lo, hi] is:
    //   [max(0, lo), hi], which is non-empty since hi >= 0.
    RealT t0 = std::max(RealT(0), lo);
    if (t0 > hi) t0 = hi; // safety, though shouldn't trigger here.

    // Map t0 (along ray) back to segment param t1.
    // Note: a0,a1 are params of endpoints; (a1 - a0) != 0 since sd2 > 0.
    const RealT t1 = clamp01((t0 - a0) / (a1 - a0));

    return {t0, t1};
  }

  default:
    return {RealT(0), RealT(0)};
  }
}

/// @ingroup core_queries
/// @brief Computes the parametric locations of closest points on a segment, and
/// a ray.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_point_parametric(const tf::segment<Dims, Policy0> &segment0,
                              const tf::ray_like<Dims, Policy1> &ray1) {
  auto out = closest_point_parametric(ray1, segment0);
  std::swap(out.first, out.second);
  return out;
}

/// @ingroup core_queries
/// @brief Computes the parametric locations of closest points on a line, and a
/// segment.
template <std::size_t Dims, typename Policy, typename Policy0>
auto closest_point_parametric(const tf::line_like<Dims, Policy> &line,
                              const tf::segment<Dims, Policy0> &segment)
    -> std::pair<tf::coordinate_type<Policy, Policy0>,
                 tf::coordinate_type<Policy, Policy0>> {

  using RealT = tf::coordinate_type<Policy, Policy0>;

  const auto &p0 = line.origin;
  const auto &d0 = line.direction;

  // Segment as a line
  auto line_segment = tf::make_line_between_points(segment[0], segment[1]);
  const auto &p1 = line_segment.origin;
  const auto &d1 = line_segment.direction;

  auto [status, t0_raw, t1_raw] =
      tf::core::line_line_check_full(line, line_segment);

  RealT t0 = t0_raw;
  RealT t1 = t1_raw;

  const RealT d0_dot = tf::dot(d0, d0);

  auto clamp01 = [](RealT v) {
    if (v < RealT(0))
      return RealT(0);
    if (v > RealT(1))
      return RealT(1);
    return v;
  };

  switch (status) {
  case tf::intersect_status::non_parallel: {
    // (t0_raw, t1_raw) = closest points on infinite lines.

    if (t1 >= RealT(0) && t1 <= RealT(1)) {
      // Already within segment; line is infinite so this is final.
      return {t0, t1};
    }

    // Otherwise clamp t1 to segment and recompute t0 from that point.
    t1 = clamp01(t1);
    if (d0_dot > RealT(0)) {
      const auto q = p1 + t1 * d1;
      t0 = tf::dot(q - p0, d0) / d0_dot;
    } else {
      // Degenerate line direction; just pick t0 = 0
      t0 = RealT(0);
    }
    return {t0, t1};
  }

  case tf::intersect_status::parallel: {
    // Parallel, non-colinear: distance from line to any point on segment is
    // constant along segment. Any endpoint works; choose segment[0] as
    // canonical.
    if (d0_dot > RealT(0)) {
      t0 = tf::dot(segment[0] - p0, d0) / d0_dot;
    } else {
      t0 = RealT(0);
    }
    t1 = RealT(0);
    return {t0, t1};
  }

  case tf::intersect_status::colinear: {
    // Colinear: segment lies on line. Infinite zero-distance solutions.
    // Canonical: use segment[0] (t1=0) and its param on the line.
    if (d0_dot > RealT(0)) {
      t0 = tf::dot(segment[0] - p0, d0) / d0_dot;
    } else {
      t0 = RealT(0);
    }
    t1 = RealT(0);
    return {t0, t1};
  }

  default:
    return {RealT(0), RealT(0)};
  }
}

/// @ingroup core_queries
/// @brief Computes the parametric locations of closest points on a segment, and
/// a line.
template <std::size_t Dims, typename Policy0, typename Policy>
auto closest_point_parametric(const tf::segment<Dims, Policy0> &segment,
                              const tf::line_like<Dims, Policy> &line) {
  auto result = closest_point_parametric(line, segment);
  return std::make_pair(result.second, result.first);
}

/// @ingroup core_queries
/// @brief Computes the parametric locations of closest points on a line, and a
/// line.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_point_parametric(const tf::line_like<Dims, Policy0> &line0,
                              const tf::line_like<Dims, Policy1> &line1)
    -> std::pair<tf::coordinate_type<Policy0, Policy1>,
                 tf::coordinate_type<Policy0, Policy1>> {
  using RealT = tf::coordinate_type<Policy0, Policy1>;

  const auto &p0 = line0.origin;
  const auto &p1 = line1.origin;
  const auto &d1 = line1.direction;

  auto [status, t0_raw, t1_raw] = tf::core::line_line_check_full(line0, line1);

  RealT t0 = t0_raw;
  RealT t1 = t1_raw;

  const RealT d1_dot = tf::dot(d1, d1);

  switch (status) {
  case tf::intersect_status::non_parallel:
    // line_line_check_full already gave the unique closest pair on infinite
    // lines
    return {t0, t1};

  case tf::intersect_status::parallel: {
    // Parallel but not colinear: choose canonical pair:
    // - fix t0 = 0 on line0 (point = p0)
    // - t1 = projection of p0 onto line1
    t0 = RealT(0);
    if (d1_dot > RealT(0)) {
      const auto diff = p0 - p1;
      t1 = tf::dot(diff, d1) / d1_dot;
    } else {
      // Degenerate line1 direction; nothing better than both zeros.
      t1 = RealT(0);
    }
    return {t0, t1};
  }

  case tf::intersect_status::colinear: {
    // Same geometric line; infinite zero-distance solutions.
    // Pick a stable representative:
    // - t0 = 0 on line0 (point = p0)
    // - t1 = projection of p0 onto line1 (if d1 non-degenerate),
    //   otherwise just 0.
    t0 = RealT(0);
    if (d1_dot > RealT(0)) {
      const auto diff = p0 - p1;
      t1 = tf::dot(diff, d1) / d1_dot;
    } else {
      t1 = RealT(0);
    }
    return {t0, t1};
  }

  default:
    return {RealT(0), RealT(0)};
  }
}

/// @ingroup core_queries
/// @brief Computes the parametric locations of closest points on a segment, and
/// a segment.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_point_parametric(const tf::segment<Dims, Policy0> &segment0,
                              const tf::segment<Dims, Policy1> &segment1)
    -> std::pair<tf::coordinate_type<Policy0, Policy1>,
                 tf::coordinate_type<Policy0, Policy1>> {
  using RealT = tf::coordinate_type<Policy0, Policy1>;

  const auto &p0 = segment0[0];
  const auto &p1 = segment0[1];
  const auto &q0 = segment1[0];
  const auto &q1 = segment1[1];

  const auto u = p1 - p0; // dir seg0
  const auto v = q1 - q0; // dir seg1

  auto line0 = tf::make_line_between_points(p0, p1);
  auto line1 = tf::make_line_between_points(q0, q1);

  auto [status, lt0, lt1] = tf::core::line_line_check_full(line0, line1);

  auto clamp01 = [](RealT x) {
    return x < RealT(0) ? RealT(0) : (x > RealT(1) ? RealT(1) : x);
  };

  auto point0 = [&](RealT t0) { return p0 + t0 * u; };
  auto point1 = [&](RealT t1) { return q0 + t1 * v; };

  auto dist2 = [&](RealT t0, RealT t1) {
    auto a = point0(t0);
    auto b = point1(t1);
    return (a - b).length2();
  };

  // Fast path: generic skew case, solution already inside both segments.
  if (status == tf::intersect_status::non_parallel && lt0 >= RealT(0) &&
      lt0 <= RealT(1) && lt1 >= RealT(0) && lt1 <= RealT(1)) {
    return {lt0, lt1};
  }

  // Slow path: handle all edge cases (including when non_parallel but outside
  // [0,1]).

  RealT best_t0 = RealT(0);
  RealT best_t1 = RealT(0);
  auto best_d2 = dist2(best_t0, best_t1);

  auto consider = [&](RealT t0, RealT t1) {
    auto d2 = dist2(t0, t1);
    if (d2 < best_d2) {
      best_d2 = d2;
      best_t0 = t0;
      best_t1 = t1;
    }
  };

  const RealT u_dot = tf::dot(u, u);
  const RealT v_dot = tf::dot(v, v);

  // 1) If we have a non-parallel analytic solution, clamp it as a candidate.
  if (status == tf::intersect_status::non_parallel) {
    consider(clamp01(lt0), clamp01(lt1));
  }

  // 2) Endpoints of seg0 projected onto seg1.
  if (v_dot > RealT(0)) {
    // p0 -> seg1
    {
      RealT t1 = tf::dot(p0 - q0, v) / v_dot;
      consider(RealT(0), clamp01(t1));
    }
    // p1 -> seg1
    {
      RealT t1 = tf::dot(p1 - q0, v) / v_dot;
      consider(RealT(1), clamp01(t1));
    }
  } else {
    // seg1 degenerate: only q0
    consider(RealT(0), RealT(0));
    consider(RealT(1), RealT(0));
  }

  // 3) Endpoints of seg1 projected onto seg0.
  if (u_dot > RealT(0)) {
    // q0 -> seg0
    {
      RealT t0 = tf::dot(q0 - p0, u) / u_dot;
      consider(clamp01(t0), RealT(0));
    }
    // q1 -> seg0
    {
      RealT t0 = tf::dot(q1 - p0, u) / u_dot;
      consider(clamp01(t0), RealT(1));
    }
  } else {
    // seg0 degenerate: only p0
    consider(RealT(0), RealT(0));
    consider(RealT(0), RealT(1));
  }

  // For parallel/colinear:
  // - These candidates cover endpoint-endpoint and endpoint-projection cases.
  // - For colinear-overlap, you'll land on one overlap endpoint (d=0, valid).

  return {best_t0, best_t1};
}

} // namespace tf
