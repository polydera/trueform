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
#include <trueform/core/algorithm/parallel_for_each.hpp>
#include <trueform/core/views/sequence_range.hpp>
#include <trueform/volume/volume_buffer.hpp>

#include <array>
#include <cmath>
#include <vector>

namespace tf {
namespace test {

/// @brief Analytic fields the volume tests and benchmarks sample.
///
/// Each shape answers `sdf(p)` in the volume's own frame and lists the sharp
/// edges it carries, so a test measures against the shape rather than against
/// a remembered number. Nothing is axis-aligned: every shape sits on one
/// generic rotation, which is what keeps a grid-aligned special case from
/// passing by accident.
namespace volume_field {

using vec3 = std::array<float, 3>;
using seg3 = std::array<vec3, 2>;

struct frame {
  std::array<std::array<float, 3>, 3> r;
  vec3 shift = {0.031f, -0.017f, 0.011f};

  frame() {
    float axis[3] = {1.f, 2.f, 3.f};
    const float len = std::sqrt(14.f);
    for (auto &a : axis)
      a /= len;
    const float c = std::cos(0.55f), s = std::sin(0.55f), t = 1.f - c;
    r = {{{t * axis[0] * axis[0] + c, t * axis[0] * axis[1] - s * axis[2],
           t * axis[0] * axis[2] + s * axis[1]},
          {t * axis[0] * axis[1] + s * axis[2], t * axis[1] * axis[1] + c,
           t * axis[1] * axis[2] - s * axis[0]},
          {t * axis[0] * axis[2] - s * axis[1],
           t * axis[1] * axis[2] + s * axis[0], t * axis[2] * axis[2] + c}}};
  }

  auto to_local(const vec3 &p) const -> vec3 {
    const vec3 q = {p[0] - shift[0], p[1] - shift[1], p[2] - shift[2]};
    return {r[0][0] * q[0] + r[1][0] * q[1] + r[2][0] * q[2],
            r[0][1] * q[0] + r[1][1] * q[1] + r[2][1] * q[2],
            r[0][2] * q[0] + r[1][2] * q[1] + r[2][2] * q[2]};
  }
  auto to_world(const vec3 &p) const -> vec3 {
    return {r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + shift[0],
            r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + shift[1],
            r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + shift[2]};
  }
};

inline auto segment_distance_2d(float px, float py, float ax, float ay,
                                float bx, float by) -> float {
  const float ex = bx - ax, ey = by - ay;
  const float wx = px - ax, wy = py - ay;
  const float ee = ex * ex + ey * ey;
  float t = ee > 0.f ? (wx * ex + wy * ey) / ee : 0.f;
  t = std::min(std::max(t, 0.f), 1.f);
  const float dx = wx - t * ex, dy = wy - t * ey;
  return std::sqrt(dx * dx + dy * dy);
}

/// @brief Signed distance to a closed 2D polygon of any winding.
inline auto polygon_sdf_2d(const std::vector<std::array<float, 2>> &poly,
                           float px, float py) -> float {
  float d = 1e30f;
  bool inside = false;
  const std::size_t n = poly.size();
  for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
    d = std::min(d, segment_distance_2d(px, py, poly[i][0], poly[i][1],
                                        poly[j][0], poly[j][1]));
    const bool crosses = (poly[i][1] > py) != (poly[j][1] > py);
    if (crosses) {
      const float xi = poly[i][0] + (py - poly[i][1]) *
                                        (poly[j][0] - poly[i][0]) /
                                        (poly[j][1] - poly[i][1]);
      if (px < xi)
        inside = !inside;
    }
  }
  return inside ? -d : d;
}

/// @brief Exact extrusion of a 2D signed distance along z with half-height h.
inline auto extrude(float d2, float z, float h) -> float {
  const float wx = d2, wy = std::abs(z) - h;
  const float ox = std::max(wx, 0.f), oy = std::max(wy, 0.f);
  return std::min(std::max(wx, wy), 0.f) + std::sqrt(ox * ox + oy * oy);
}

/// @brief Exact distance to an axis-aligned box of half-extents @p h.
inline auto box_sdf(const vec3 &q, const vec3 &h) -> float {
  const float dx = std::abs(q[0]) - h[0];
  const float dy = std::abs(q[1]) - h[1];
  const float dz = std::abs(q[2]) - h[2];
  const float ox = std::max(dx, 0.f), oy = std::max(dy, 0.f),
              oz = std::max(dz, 0.f);
  return std::sqrt(ox * ox + oy * oy + oz * oz) +
         std::min(std::max(dx, std::max(dy, dz)), 0.f);
}

/// @brief Exact distance to a z-axial cylinder of radius @p r, half-height @p h.
inline auto capped_cylinder_sdf(const vec3 &q, float r, float h) -> float {
  const float dr = std::sqrt(q[0] * q[0] + q[1] * q[1]) - r;
  const float dz = std::abs(q[2]) - h;
  const float ox = std::max(dr, 0.f), oz = std::max(dz, 0.f);
  return std::min(std::max(dr, dz), 0.f) + std::sqrt(ox * ox + oz * oz);
}

/// @brief The circle of radius @p r at height @p z, as a closed polyline.
inline auto circle_arcs(const frame &f, float r, float z, int segments)
    -> std::vector<seg3> {
  std::vector<seg3> out;
  const float step = 2.f * 3.14159265358979f / float(segments);
  for (int i = 0; i < segments; ++i) {
    const float a0 = float(i) * step, a1 = float(i + 1) * step;
    out.push_back({f.to_world({r * std::cos(a0), r * std::sin(a0), z}),
                   f.to_world({r * std::cos(a1), r * std::sin(a1), z})});
  }
  return out;
}

/// @brief The smooth control: no crease anywhere.
struct sphere {
  float radius = 0.7f;
  auto sdf(const vec3 &p) const -> float {
    return std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) - radius;
  }
  auto edges() const -> std::vector<seg3> { return {}; }
  auto corners() const -> std::vector<vec3> { return {}; }
};

/// @brief A tilted box: three face orientations meeting at 90 degree creases.
struct box {
  frame f;
  vec3 half = {0.62f, 0.45f, 0.33f};

  auto sdf(const vec3 &p) const -> float {
    return box_sdf(f.to_local(p), half);
  }
  auto edges() const -> std::vector<seg3> {
    std::vector<seg3> out;
    auto corner = [&](int sx, int sy, int sz) {
      return f.to_world({sx * half[0], sy * half[1], sz * half[2]});
    };
    const int signs[2] = {-1, 1};
    for (int a : signs)
      for (int b : signs) {
        out.push_back({corner(-1, a, b), corner(1, a, b)});
        out.push_back({corner(a, -1, b), corner(a, 1, b)});
        out.push_back({corner(a, b, -1), corner(a, b, 1)});
      }
    return out;
  }
  /// @brief The eight three-plane corners.
  auto corners() const -> std::vector<vec3> {
    std::vector<vec3> out;
    const int signs[2] = {-1, 1};
    for (int a : signs)
      for (int b : signs)
        for (int c : signs)
          out.push_back(f.to_world({a * half[0], b * half[1], c * half[2]}));
    return out;
  }
};

/// @brief A tilted regular octahedron: twelve creases of dihedral
/// `arccos(-1/3)` meeting at six corners where FOUR planes do.
struct octahedron {
  frame f;
  float size = 0.72f;

  auto sdf(const vec3 &p) const -> float {
    const auto q = f.to_local(p);
    const float a[3] = {std::abs(q[0]), std::abs(q[1]), std::abs(q[2])};
    const float m = a[0] + a[1] + a[2] - size;
    float r[3];
    if (3.f * a[0] < m)
      r[0] = a[0], r[1] = a[1], r[2] = a[2];
    else if (3.f * a[1] < m)
      r[0] = a[1], r[1] = a[2], r[2] = a[0];
    else if (3.f * a[2] < m)
      r[0] = a[2], r[1] = a[0], r[2] = a[1];
    else
      return m * 0.5773502692f;
    const float k = std::min(std::max(0.5f * (r[2] - r[1] + size), 0.f), size);
    const float dy = r[1] - size + k, dz = r[2] - k;
    return std::sqrt(r[0] * r[0] + dy * dy + dz * dz);
  }
  auto vertices() const -> std::vector<vec3> {
    std::vector<vec3> out;
    for (int axis = 0; axis < 3; ++axis)
      for (int s : {-1, 1}) {
        vec3 v{0.f, 0.f, 0.f};
        v[std::size_t(axis)] = float(s) * size;
        out.push_back(f.to_world(v));
      }
    return out;
  }
  auto edges() const -> std::vector<seg3> {
    const auto v = vertices();
    std::vector<seg3> out;
    for (std::size_t i = 0; i < v.size(); ++i)
      for (std::size_t j = i + 1; j < v.size(); ++j)
        if (i / 2 != j / 2)
          out.push_back({v[i], v[j]});
    return out;
  }
  /// @brief The six four-plane corners.
  auto corners() const -> std::vector<vec3> { return vertices(); }
};

/// @brief A tilted capped cylinder: two circular creases, curved and convex.
struct cylinder {
  frame f;
  float radius = 0.55f;
  float half_z = 0.42f;

  auto sdf(const vec3 &p) const -> float {
    return capped_cylinder_sdf(f.to_local(p), radius, half_z);
  }
  auto edges() const -> std::vector<seg3> {
    auto out = circle_arcs(f, radius, -half_z, 256);
    const auto top = circle_arcs(f, radius, half_z, 256);
    out.insert(out.end(), top.begin(), top.end());
    return out;
  }
  auto corners() const -> std::vector<vec3> { return {}; }
};

/// @brief A tilted box bored through by a cylinder: the bore's two mouths are
/// curved CONCAVE creases, the box's twelve edges straight convex ones.
///
/// The field is the difference of two exact distances, which is the shape's own
/// distance on the side the difference leaves piecewise affine — the same
/// contract @ref gear states.
struct box_cylinder {
  frame f;
  vec3 half = {0.62f, 0.62f, 0.34f};
  float radius = 0.30f;

  auto sdf(const vec3 &p) const -> float {
    const auto q = f.to_local(p);
    return std::max(box_sdf(q, half),
                    -capped_cylinder_sdf(q, radius, 2.f * half[2]));
  }
  auto edges() const -> std::vector<seg3> {
    std::vector<seg3> out;
    auto corner = [&](int sx, int sy, int sz) {
      return f.to_world({sx * half[0], sy * half[1], sz * half[2]});
    };
    const int signs[2] = {-1, 1};
    for (int a : signs)
      for (int b : signs) {
        out.push_back({corner(-1, a, b), corner(1, a, b)});
        out.push_back({corner(a, -1, b), corner(a, 1, b)});
        out.push_back({corner(a, b, -1), corner(a, b, 1)});
      }
    for (int s : signs) {
      const auto mouth = circle_arcs(f, radius, float(s) * half[2], 256);
      out.insert(out.end(), mouth.begin(), mouth.end());
    }
    return out;
  }
  auto corners() const -> std::vector<vec3> {
    std::vector<vec3> out;
    const int signs[2] = {-1, 1};
    for (int a : signs)
      for (int b : signs)
        for (int c : signs)
          out.push_back(f.to_world({a * half[0], b * half[1], c * half[2]}));
    return out;
  }
};

/// @brief A tilted capped cone of half-angle `angle_deg`: one apex the true
/// vertex of which sits further outside its cell the sharper the cone is, and
/// one circular rim.
struct cone {
  frame f;
  float angle_deg;
  float height = 0.9f;
  float apex_z = 0.45f;

  explicit cone(float angle = 30.f) : angle_deg(angle) {}

  auto base_radius() const -> float {
    return height * std::tan(angle_deg * 3.14159265358979f / 180.f);
  }
  auto sdf(const vec3 &p) const -> float {
    const auto l = f.to_local(p);
    // the profile in (radial, axial): apex at the origin, rim at q
    const float qx = base_radius(), qy = -height;
    const float wx = std::sqrt(l[0] * l[0] + l[1] * l[1]), wy = l[2] - apex_z;
    const float qq = qx * qx + qy * qy;
    float t = (wx * qx + wy * qy) / qq;
    t = std::min(std::max(t, 0.f), 1.f);
    const float ax = wx - t * qx, ay = wy - t * qy;
    float u = qx > 0.f ? wx / qx : 0.f;
    u = std::min(std::max(u, 0.f), 1.f);
    const float bx = wx - u * qx, by = wy - qy;
    const float d2 = std::min(ax * ax + ay * ay, bx * bx + by * by);
    const float s = std::max(-(wx * qy - wy * qx), -(wy - qy));
    return std::sqrt(d2) * (s < 0.f ? -1.f : 1.f);
  }
  auto edges() const -> std::vector<seg3> {
    return circle_arcs(f, base_radius(), apex_z - height, 256);
  }
  /// @brief The apex: the corner whose distance from its cell grows as the
  /// half-angle shrinks.
  auto corners() const -> std::vector<vec3> {
    return {f.to_world({0.f, 0.f, apex_z})};
  }
};

/// @brief A tilted box whose edges carry a fillet of radius @p radius: the
/// fidelity control. At radius zero it is a box and must sharpen; at a radius
/// of two cells and more it is smooth and must NOT.
struct rounded_box {
  frame f;
  float radius;
  vec3 half = {0.62f, 0.45f, 0.33f};

  explicit rounded_box(float r = 0.f) : radius(r) {}

  auto sdf(const vec3 &p) const -> float {
    const vec3 inner = {half[0] - radius, half[1] - radius, half[2] - radius};
    return box_sdf(f.to_local(p), inner) - radius;
  }
  /// @brief The creases, which exist only where the fillet does not.
  auto edges() const -> std::vector<seg3> {
    if (radius > 0.f)
      return {};
    std::vector<seg3> out;
    auto corner = [&](int sx, int sy, int sz) {
      return f.to_world({sx * half[0], sy * half[1], sz * half[2]});
    };
    const int signs[2] = {-1, 1};
    for (int a : signs)
      for (int b : signs) {
        out.push_back({corner(-1, a, b), corner(1, a, b)});
        out.push_back({corner(a, -1, b), corner(a, 1, b)});
        out.push_back({corner(a, b, -1), corner(a, b, 1)});
      }
    return out;
  }
  auto corners() const -> std::vector<vec3> {
    if (radius > 0.f)
      return {};
    std::vector<vec3> out;
    const int signs[2] = {-1, 1};
    for (int a : signs)
      for (int b : signs)
        for (int c : signs)
          out.push_back(f.to_world({a * half[0], b * half[1], c * half[2]}));
    return out;
  }
};

/// @brief A tilted triangular prism whose apex dihedral is `angle_deg`. The
/// reported edge is the apex alone, so a metric over it isolates that angle.
struct wedge {
  frame f;
  float angle_deg;
  float length = 0.62f;
  float half_z = 0.55f;
  std::vector<std::array<float, 2>> poly;

  explicit wedge(float angle = 90.f) : angle_deg(angle) {
    const float a = 0.5f * angle_deg * 3.14159265358979f / 180.f;
    const float w = length * std::tan(a);
    poly = {{-0.5f * length, 0.f}, {0.5f * length, w}, {0.5f * length, -w}};
  }
  auto sdf(const vec3 &p) const -> float {
    const auto q = f.to_local(p);
    return extrude(polygon_sdf_2d(poly, q[0], q[1]), q[2], half_z);
  }
  auto edges() const -> std::vector<seg3> {
    return {{f.to_world({poly[0][0], poly[0][1], -half_z}),
             f.to_world({poly[0][0], poly[0][1], half_z})}};
  }
  auto corners() const -> std::vector<vec3> { return {}; }
};

/// @brief An extruded toothed profile minus a bore: many convex and concave
/// creases close together.
struct gear {
  frame f;
  int teeth = 8;
  float root = 0.42f;
  float tip = 0.68f;
  float bore = 0.17f;
  float half_z = 0.30f;
  std::vector<std::array<float, 2>> poly;

  gear() {
    const float step = 2.f * 3.14159265358979f / float(teeth);
    for (int i = 0; i < teeth; ++i) {
      const float a0 = float(i) * step;
      const float ang[4] = {a0, a0 + 0.25f * step, a0 + 0.5f * step,
                            a0 + 0.75f * step};
      const float rad[4] = {root, tip, tip, root};
      for (int k = 0; k < 4; ++k)
        poly.push_back({rad[k] * std::cos(ang[k]), rad[k] * std::sin(ang[k])});
    }
  }
  auto sdf(const vec3 &p) const -> float {
    const auto q = f.to_local(p);
    const float body = extrude(polygon_sdf_2d(poly, q[0], q[1]), q[2], half_z);
    const float hole = std::sqrt(q[0] * q[0] + q[1] * q[1]) - bore;
    return std::max(body, -hole);
  }
  /// @brief Every crease the shape carries: the profile's vertical edges, the
  /// two rims where the walls meet the caps, and the bore's two mouths.
  auto edges() const -> std::vector<seg3> {
    std::vector<seg3> out;
    for (std::size_t i = 0; i < poly.size(); ++i) {
      const auto &v = poly[i];
      const auto &w = poly[(i + 1) % poly.size()];
      out.push_back({f.to_world({v[0], v[1], -half_z}),
                     f.to_world({v[0], v[1], half_z})});
      for (int s : {-1, 1})
        out.push_back({f.to_world({v[0], v[1], float(s) * half_z}),
                       f.to_world({w[0], w[1], float(s) * half_z})});
    }
    for (int s : {-1, 1}) {
      const auto mouth = circle_arcs(f, bore, float(s) * half_z, 128);
      out.insert(out.end(), mouth.begin(), mouth.end());
    }
    return out;
  }
  auto corners() const -> std::vector<vec3> { return {}; }
};

/// @brief A tilted slab, thinner than a voxel at moderate resolutions: the
/// topology stress. Unbounded, so its surface leaves the domain.
struct slab {
  frame f;
  float thickness = 0.012f;
  auto sdf(const vec3 &p) const -> float {
    const auto q = f.to_local(p);
    return std::abs(q[2]) - 0.5f * thickness;
  }
  auto edges() const -> std::vector<seg3> { return {}; }
  auto corners() const -> std::vector<vec3> { return {}; }
};

/// @brief A closed slab: the same sheet, capped inside the domain.
struct bounded_slab {
  frame f;
  float thickness = 0.012f;
  float half = 0.8f;
  auto sdf(const vec3 &p) const -> float {
    const auto q = f.to_local(p);
    const float sheet = std::abs(q[2]) - 0.5f * thickness;
    const float extent = std::max(std::abs(q[0]) - half, std::abs(q[1]) - half);
    return std::max(sheet, extent);
  }
  auto edges() const -> std::vector<seg3> { return {}; }
  auto corners() const -> std::vector<vec3> { return {}; }
};

/// @brief A gyroid: not a distance field, and dense in triangles.
struct gyroid {
  float periods = 3.f;
  auto sdf(const vec3 &p) const -> float {
    const float s = periods * 3.14159265358979f;
    const float x = p[0] * s, y = p[1] * s, z = p[2] * s;
    return std::sin(x) * std::cos(y) + std::sin(y) * std::cos(z) +
           std::sin(z) * std::cos(x);
  }
  auto edges() const -> std::vector<seg3> { return {}; }
  auto corners() const -> std::vector<vec3> { return {}; }
};

/// @brief A closed gyroid: the same field, intersected with a ball.
struct bounded_gyroid {
  gyroid field;
  float radius = 0.9f;
  auto sdf(const vec3 &p) const -> float {
    const float ball =
        std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) - radius;
    return std::max(field.sdf(p), ball);
  }
  auto edges() const -> std::vector<seg3> { return {}; }
  auto corners() const -> std::vector<vec3> { return {}; }
};

/// @brief A deterministic pseudo-random sign field: the densest source of
/// marching-cubes cases and of ambiguous neighbourhoods.
struct noise {
  float cell = 12.f;
  auto sdf(const vec3 &p) const -> float {
    const auto h = [](int a, int b, int c) {
      unsigned x = unsigned(a) * 73856093u ^ unsigned(b) * 19349663u ^
                   unsigned(c) * 83492791u;
      x ^= x >> 13;
      x *= 1274126177u;
      x ^= x >> 16;
      return x;
    };
    const int i = int(std::floor(p[0] * cell)) + 512;
    const int j = int(std::floor(p[1] * cell)) + 512;
    const int k = int(std::floor(p[2] * cell)) + 512;
    return (h(i, j, k) & 1u) ? 1.f : -1.f;
  }
  auto edges() const -> std::vector<seg3> { return {}; }
  auto corners() const -> std::vector<vec3> { return {}; }
};

/// @brief The field's normal at a point, by central differences.
template <typename Field>
auto field_gradient(const Field &field, const vec3 &p, float h = 1e-3f)
    -> vec3 {
  vec3 g{};
  for (int d = 0; d < 3; ++d) {
    vec3 lo = p, hi = p;
    lo[std::size_t(d)] -= h;
    hi[std::size_t(d)] += h;
    g[std::size_t(d)] = (field.sdf(hi) - field.sdf(lo)) / (2.f * h);
  }
  const float l = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
  if (l > 0.f)
    for (auto &v : g)
      v /= l;
  return g;
}

} // namespace volume_field

/// @brief Sample a field onto a cubic grid centred on the origin.
///
/// @tparam T The sample type the grid stores, which is its coordinate type too.
/// @param n Samples per axis.
/// @param extent The grid's span along each axis.
template <typename T = float, typename Field>
auto sampled_volume(const Field &field, int n, float extent = 2.4f)
    -> tf::volume_buffer<T> {
  const float spacing = n > 1 ? extent / float(n - 1) : extent;
  const float org = -extent / 2;
  tf::volume_buffer<T> vol({n, n, n}, {T(spacing), T(spacing), T(spacing)},
                           {T(org), T(org), T(org)});
  tf::parallel_for_each(tf::make_sequence_range(0, n), [&](int z) {
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x)
        vol.samples_buffer()[vol.linear_index(x, y, z)] = T(field.sdf(
            {org + x * spacing, org + y * spacing, org + z * spacing}));
  });
  return vol;
}

} // namespace test
} // namespace tf
