#include "../common/volume_generators.hpp"

#include <catch2/catch_test_macros.hpp>

#include <trueform/core/distance.hpp>
#include <trueform/volume/impl/dual_contour_planes.hpp>
#include <trueform/volume/impl/flying_dc.hpp>
#include <trueform/volume/make_isosurface.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

namespace sharp_field = tf::test::volume_field;
using tf::volume_detail::refit_state;

struct sharp_extraction {
  tf::polygons_buffer<int, float, 3, 3> mesh;
  tf::buffer<tf::volume_detail::refit_record> provenance;
  float spacing = 0.f;
};

template <typename Field>
auto sharp_extract(const Field &field, int n, bool refine) -> sharp_extraction {
  const auto vol = tf::test::sampled_volume(field, n);
  tf::isosurface_config config;
  config.method = tf::isosurface_method::dual_contouring;
  config.refine = refine;
  tf::volume_detail::dual_contouring<int, float> dc;
  dc.record_refit_provenance(true);
  sharp_extraction out;
  out.mesh = dc.build(vol.volume(), 0.f, config);
  out.provenance.allocate(dc.refit_provenance().size());
  std::copy(dc.refit_provenance().begin(), dc.refit_provenance().end(),
            out.provenance.begin());
  out.spacing = float(vol.spacing()[0]);
  return out;
}

// The distance from a densely sampled analytic crease to the extracted surface,
// in units of the grid spacing.
template <typename Polygons>
auto sharp_crease_chamfer(const Polygons &polygons,
                          const std::vector<sharp_field::seg3> &edges,
                          float spacing, int per_edge) -> double {
  double sum = 0;
  std::size_t count = 0;
  for (const auto &e : edges)
    for (int k = 0; k < per_edge; ++k) {
      const float t = (k + 0.5f) / float(per_edge);
      const auto p = tf::make_point(e[0][0] + t * (e[1][0] - e[0][0]),
                                    e[0][1] + t * (e[1][1] - e[0][1]),
                                    e[0][2] + t * (e[1][2] - e[0][2]));
      double best = 1e30;
      for (const auto &poly : polygons)
        best = std::min(best, double(tf::distance2(poly, p)));
      sum += std::sqrt(best);
      ++count;
    }
  return sum / double(count) / double(spacing);
}

auto sharp_moved_fraction(const tf::buffer<tf::volume_detail::refit_record> &prov,
                          float spacing) -> double {
  std::size_t moved = 0;
  for (std::size_t i = 0; i < prov.size(); ++i)
    if (prov[i].moved > 0.05f * spacing)
      ++moved;
  return prov.size() > 0 ? double(moved) / double(prov.size()) : 0.0;
}

/// The distance from a point to the nearest of a shape's analytic creases.
auto sharp_crease_reach(const std::vector<sharp_field::seg3> &edges, float px,
                        float py, float pz) -> double {
  double best = 1e30;
  for (const auto &e : edges) {
    const double ex = e[1][0] - e[0][0], ey = e[1][1] - e[0][1],
                 ez = e[1][2] - e[0][2];
    const double wx = px - e[0][0], wy = py - e[0][1], wz = pz - e[0][2];
    const double ee = ex * ex + ey * ey + ez * ez;
    double t = ee > 0 ? (wx * ex + wy * ey + wz * ez) / ee : 0.0;
    t = std::min(std::max(t, 0.0), 1.0);
    const double dx = wx - t * ex, dy = wy - t * ey, dz = wz - t * ez;
    best = std::min(best, std::sqrt(dx * dx + dy * dy + dz * dz));
  }
  return best;
}

/// Per extraction: triangles whose area vanishes, and triangles whose normal
/// opposes the field's own gradient well away from every analytic crease.
struct sharp_triangle_report {
  std::size_t degenerate = 0;
  std::size_t inverted_far = 0;
  std::size_t n_far = 0;
};

template <typename Field, typename Polygons>
auto sharp_inspect_triangles(const Field &field, const Polygons &polygons,
                             float spacing) -> sharp_triangle_report {
  sharp_triangle_report r;
  const auto edges = field.edges();
  const double tiny = 1e-6 * double(spacing) * double(spacing);
  for (const auto &tri : polygons) {
    const auto a = tri[0], b = tri[1], c = tri[2];
    const double u[3] = {double(b[0]) - a[0], double(b[1]) - a[1],
                         double(b[2]) - a[2]};
    const double v[3] = {double(c[0]) - a[0], double(c[1]) - a[1],
                         double(c[2]) - a[2]};
    const double nx = u[1] * v[2] - u[2] * v[1];
    const double ny = u[2] * v[0] - u[0] * v[2];
    const double nz = u[0] * v[1] - u[1] * v[0];
    const double area = 0.5 * std::sqrt(nx * nx + ny * ny + nz * nz);
    if (area < tiny) {
      ++r.degenerate;
      continue;
    }
    const sharp_field::vec3 centroid = {
        float((double(a[0]) + b[0] + c[0]) / 3.0),
        float((double(a[1]) + b[1] + c[1]) / 3.0),
        float((double(a[2]) + b[2] + c[2]) / 3.0)};
    if (sharp_crease_reach(edges, centroid[0], centroid[1], centroid[2]) <
        1.5 * double(spacing))
      continue;
    ++r.n_far;
    const auto g = sharp_field::field_gradient(field, centroid);
    if (nx * g[0] + ny * g[1] + nz * g[2] <= 0.0)
      ++r.inverted_far;
  }
  return r;
}

} // namespace

// The fixtures are the measurement, so their own claims are checked first: an
// exact distance field carries a unit gradient away from its features, and the
// features it reports lie on its surface.
TEST_CASE("the sharpness fixtures are exact distance fields",
          "[volume][sharpness]") {
  const auto check = [](const auto &field) {
    for (const auto &e : field.edges())
      for (int k = 0; k <= 2; ++k) {
        const float t = 0.5f * float(k);
        const sharp_field::vec3 p = {e[0][0] + t * (e[1][0] - e[0][0]),
                                     e[0][1] + t * (e[1][1] - e[0][1]),
                                     e[0][2] + t * (e[1][2] - e[0][2])};
        REQUIRE(std::abs(field.sdf(p)) < 1e-4f);
      }
    for (const auto &c : field.corners())
      REQUIRE(std::abs(field.sdf(c)) < 1e-4f);
    // a point well clear of every feature: the gradient of a distance field is
    // a unit vector there
    const sharp_field::vec3 probe = {0.83f, -0.91f, 0.77f};
    const auto g = sharp_field::field_gradient(field, probe);
    REQUIRE(std::abs(std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) - 1.f) <
            1e-2f);
  };
  check(sharp_field::box{});
  check(sharp_field::octahedron{});
  check(sharp_field::cylinder{});
  check(sharp_field::cone(30.f));
  check(sharp_field::rounded_box(0.05f));
}

// The claim the refit stands on: a distance field is exactly `n . p + d` inside
// a face's nearest-feature region, so the samples of that region determine the
// plane. The seeds here are deliberately wrong by several degrees — what the
// regression has to recover from is the identification, not the estimate.
TEST_CASE("the samples state a plane the seed only approximates",
          "[volume][sharpness]") {
  using namespace tf::volume_detail;
  const sharp_field::box shape;
  const int n = 64;
  const auto vol = tf::test::sampled_volume(shape, n);
  dc_grid g;
  g.nx = g.ny = g.nz = n;
  g.cx = g.cy = g.cz = n - 1;
  for (int i = 0; i < 3; ++i) {
    g.origin[i] = double(vol.origin()[i]);
    g.spacing[i] = double(vol.spacing()[i]);
  }
  const double h = g.spacing[0];

  // the two faces meeting on the box's first analytic edge
  const auto face_normal = [&](int axis) {
    sharp_field::vec3 a{0.f, 0.f, 0.f};
    a[std::size_t(axis)] = -1.f;
    const auto wa = shape.f.to_world(a);
    const auto wb = shape.f.to_world({0.f, 0.f, 0.f});
    sharp_field::vec3 nrm{wa[0] - wb[0], wa[1] - wb[1], wa[2] - wb[2]};
    const float l =
        std::sqrt(nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]);
    for (auto &v : nrm)
      v /= l;
    return nrm;
  };
  const auto on_edge =
      shape.f.to_world({0.f, -shape.half[1], -shape.half[2]});

  const auto box_edges = shape.edges();
  const auto &e = box_edges[0];
  int tested = 0, recovered = 0;
  for (int k = 8; k < 56; k += 4) {
    const float t = (k + 0.5f) / 64.f;
    const sharp_field::vec3 p = {e[0][0] + t * (e[1][0] - e[0][0]),
                                 e[0][1] + t * (e[1][1] - e[0][1]),
                                 e[0][2] + t * (e[1][2] - e[0][2])};
    const int cx = int((double(p[0]) - g.origin[0]) / h);
    const int cy = int((double(p[1]) - g.origin[1]) / h);
    const int cz = int((double(p[2]) - g.origin[2]) / h);
    tf::point<double, 3> sp[k_pool_sample_cap];
    double sv[k_pool_sample_cap];
    const int m = gather_pool_samples<double>(
        g, make_field_samples<double>(vol.volume()), 0.0, cx, cy, cz, sp, sv);
    REQUIRE(m == k_pool_sample_cap);

    double seed_n[k_plane_cap][3]{};
    double seed_d[k_plane_cap]{};
    bool seed_ok[k_plane_cap] = {true, true, false};
    const int axes[2] = {1, 2};
    for (int s = 0; s < 2; ++s) {
      const auto exact = face_normal(axes[s]);
      // tilt the seed by about four degrees about z
      const double c = std::cos(0.07), si = std::sin(0.07);
      const double nn[3] = {exact[0] * c - exact[1] * si,
                            exact[0] * si + exact[1] * c, exact[2]};
      for (int d = 0; d < 3; ++d)
        seed_n[s][d] = nn[d];
      seed_d[s] = -(nn[0] * on_edge[0] + nn[1] * on_edge[1] + nn[2] * on_edge[2]);
    }
    pool_plane planes[k_plane_cap]{};
    const int certified = regress_pool_planes(sp, sv, m, h, seed_n, seed_d,
                                              seed_ok, planes);
    ++tested;
    if (certified < 2)
      continue;
    ++recovered;
    // every certified plane IS one of the two faces: its normal and its
    // offset, not the seed's. A certificate that could be earned by a plane
    // the region does not carry would state nothing.
    for (int j = 0; j < k_plane_cap; ++j) {
      if (!planes[j].certified)
        continue;
      const double len = std::sqrt(planes[j].n[0] * planes[j].n[0] +
                                   planes[j].n[1] * planes[j].n[1] +
                                   planes[j].n[2] * planes[j].n[2]);
      double best_dot = -2.0, best_offset = 1e30;
      for (int s = 0; s < 2; ++s) {
        const auto exact = face_normal(axes[s]);
        const double dot = (planes[j].n[0] * exact[0] +
                            planes[j].n[1] * exact[1] +
                            planes[j].n[2] * exact[2]) /
                           len;
        if (dot > best_dot)
          best_dot = dot;
      }
      best_offset = std::abs(planes[j].n[0] * on_edge[0] +
                             planes[j].n[1] * on_edge[1] +
                             planes[j].n[2] * on_edge[2] + planes[j].d) /
                    len;
      // the gradient of a distance field is a unit vector
      INFO("cell " << cx << " " << cy << " " << cz << " plane " << j
                   << " dot " << best_dot << " offset " << best_offset / h
                   << " h");
      REQUIRE(std::abs(len - 1.0) < 0.02);
      REQUIRE(best_dot > 0.99);
      REQUIRE(best_offset < 0.12 * h);
    }
  }
  REQUIRE(tested > 8);
  // and the seed's four degrees do not stop the samples from stating both
  REQUIRE(recovered * 3 > tested);
}

// One plane stated twice is one plane. A cluster split across a face seeds two
// slots that converge on the same plane, and the copy would both occupy the
// slot a missing face needs and take crossings whose own face then never enters
// the quadric.
TEST_CASE("the same plane stated twice comes back once",
          "[volume][sharpness]") {
  using namespace tf::volume_detail;
  // a 4x4x4 block of samples of an exactly affine field: one plane and nothing
  // else, so anything a fit certifies here is that plane
  const double h = 0.1;
  const double truth[3] = {0.6, 0.48, 0.64};
  const double truth_d = -0.15;
  tf::point<double, 3> sample_p[k_pool_sample_cap];
  double sample_v[k_pool_sample_cap];
  int m = 0;
  for (int k = 0; k < 4; ++k)
    for (int j = 0; j < 4; ++j)
      for (int i = 0; i < 4; ++i) {
        sample_p[m] = tf::make_point((i - 1.5) * h, (j - 1.5) * h,
                                     (k - 1.5) * h);
        sample_v[m] = truth[0] * sample_p[m][0] + truth[1] * sample_p[m][1] +
                      truth[2] * sample_p[m][2] + truth_d;
        ++m;
      }
  REQUIRE(m == k_pool_sample_cap);

  // two seeds of that one plane, tilted apart about z so each takes the half of
  // the block nearer to it
  double seed_n[k_plane_cap][3]{};
  double seed_d[k_plane_cap]{};
  bool seed_ok[k_plane_cap] = {true, true, false};
  for (int s = 0; s < 2; ++s) {
    const double a = (s == 0 ? 0.035 : -0.035);
    const double c = std::cos(a), si = std::sin(a);
    seed_n[s][0] = truth[0] * c - truth[1] * si;
    seed_n[s][1] = truth[0] * si + truth[1] * c;
    seed_n[s][2] = truth[2];
    seed_d[s] = truth_d;
  }

  pool_plane planes[k_plane_cap]{};
  const int certified =
      state_pool_planes(sample_p, sample_v, m, nullptr, nullptr, 0, h, seed_n,
                        seed_d, seed_ok, planes);
  REQUIRE(certified == 1);
  int live = 0;
  for (int j = 0; j < k_plane_cap; ++j)
    live += planes[j].certified;
  REQUIRE(live == 1);
  for (int j = 0; j < k_plane_cap; ++j) {
    if (!planes[j].certified)
      continue;
    const double len = planes[j].gradient;
    REQUIRE(std::abs(len - 1.0) < 0.02);
    REQUIRE((planes[j].n[0] * truth[0] + planes[j].n[1] * truth[1] +
             planes[j].n[2] * truth[2]) /
                len >
            0.999);
  }

  // and the seeds are the caller's: a call states planes, never rewrites the
  // clustering its caller's crossings were assigned against
  for (int s = 0; s < 2; ++s) {
    const double a = (s == 0 ? 0.035 : -0.035);
    const double c = std::cos(a), si = std::sin(a);
    REQUIRE(seed_n[s][0] == truth[0] * c - truth[1] * si);
    REQUIRE(seed_n[s][1] == truth[0] * si + truth[1] * c);
    REQUIRE(seed_n[s][2] == truth[2]);
    REQUIRE(seed_d[s] == truth_d);
    REQUIRE(seed_ok[s]);
  }
  REQUIRE_FALSE(seed_ok[2]);
}

// A sample's value is the distance to the nearest surface point, so nothing on
// the surface is closer to it than that. That inequality is the whole of the
// test, and it is what tells a crease from the phantom corner a fillet's two
// flanks meet at.
TEST_CASE("the samples refuse a point no surface can reach",
          "[volume][sharpness]") {
  using namespace tf::volume_detail;
  // the unit sphere, sampled at its centre and along its axes at radius two
  tf::point<double, 3> p[7];
  double v[7];
  int n = 0;
  p[n] = tf::make_point(0.0, 0.0, 0.0);
  v[n++] = -1.0;
  for (int axis = 0; axis < 3; ++axis)
    for (int s : {-1, 1}) {
      double q[3] = {0, 0, 0};
      q[axis] = 2.0 * s;
      p[n] = tf::make_point(q[0], q[1], q[2]);
      v[n++] = 1.0;
    }
  const double on_sphere[3] = {1.0, 0.0, 0.0};
  const double centre[3] = {0.0, 0.0, 0.0};
  REQUIRE(tangency_admits(p, v, n, on_sphere, 1.0, 0.05));
  // the centre's own sample states a surface a unit away, and a surface point
  // AT the centre is none
  REQUIRE(!tangency_admits(p, v, n, centre, 1.0, 0.05));
  // the gradient is what turns a field value into a length: state a steep one
  // and the same values put the surface within the tolerance of the centre
  REQUIRE(tangency_admits(p, v, n, centre, 40.0, 0.05));
}

// The end-to-end claim, against the shape rather than a remembered number: the
// refit puts a crease vertex on the crease, and flying edges cannot.
TEST_CASE("the refit lands the crease on the crease", "[volume][sharpness]") {
  const sharp_field::box shape;
  const int n = 48;
  const auto refined = sharp_extract(shape, n, true);
  const auto plain = sharp_extract(shape, n, false);
  const auto vol = tf::test::sampled_volume(shape, n);
  const auto fe = tf::make_isosurface(vol.volume(), 0.f);
  const auto edges = shape.edges();

  const double sharp =
      sharp_crease_chamfer(refined.mesh.polygons(), edges, refined.spacing, 8);
  const double dull =
      sharp_crease_chamfer(plain.mesh.polygons(), edges, plain.spacing, 8);
  const double grid = sharp_crease_chamfer(fe.polygons(), edges, refined.spacing, 8);

  INFO("refit " << sharp << " h, unrefined " << dull << " h, flying edges "
                << grid << " h");
  REQUIRE(sharp < 0.08);
  REQUIRE(sharp * 3.0 < dull);
  REQUIRE(sharp * 6.0 < grid);

  // and the provenance says so: the crease vertices are certified refits
  std::size_t certified = 0, refits = 0;
  for (std::size_t i = 0; i < refined.provenance.size(); ++i) {
    refits += refined.provenance[i].state == refit_state::refit;
    certified += refined.provenance[i].certified;
  }
  REQUIRE(refits > 0);
  REQUIRE(certified * 2 > refits);
}

// The fidelity side of the same mechanism: a fillet's two flanks are just as
// certified as a crease's two faces, and only the samples tell them apart. A
// smooth shape must come out of the refit unmoved.
TEST_CASE("the refit does not crease a fillet", "[volume][sharpness]") {
  const int n = 64;
  const float spacing = 2.4f / float(n - 1);
  {
    const auto smooth = sharp_extract(sharp_field::sphere{}, n, true);
    REQUIRE(sharp_moved_fraction(smooth.provenance, smooth.spacing) == 0.0);
  }
  {
    const auto fillet =
        sharp_extract(sharp_field::rounded_box(4.f * spacing), n, true);
    REQUIRE(sharp_moved_fraction(fillet.provenance, fillet.spacing) < 0.2);
  }
  {
    const auto fillet =
        sharp_extract(sharp_field::rounded_box(2.f * spacing), n, true);
    REQUIRE(sharp_moved_fraction(fillet.provenance, fillet.spacing) < 0.14);
  }
  {
    // and a true crease still moves: the gate is fidelity, not silence
    const auto sharp_box =
        sharp_extract(sharp_field::rounded_box(0.f), n, true);
    REQUIRE(sharp_moved_fraction(sharp_box.provenance, sharp_box.spacing) >
            0.05);
  }
}

// The law the refinement states about itself: only a refit moves a vertex.
// Every other verdict — the samples refusing the point, the planes stating no
// feature, a solve that left its cells or did not solve at all — leaves the
// vertex the cell's own pass gave it. The neighbourhood centroid a pooled solve
// starts from is not an answer about a cell.
TEST_CASE("only a refit moves a vertex", "[volume][sharpness]") {
  // The gear is the fixture whose crowded creases reach every verdict: its
  // pools solve outside the cells they are allowed (`clamped`) as well as
  // refusing on the samples and on the plane count.
  const int n = 48;
  const auto refined = sharp_extract(sharp_field::gear{}, n, true);
  const auto plain = sharp_extract(sharp_field::gear{}, n, false);
  REQUIRE(refined.mesh.points().size() == plain.mesh.points().size());
  REQUIRE(refined.provenance.size() > 0);

  std::size_t refits = 0, clamped = 0;
  for (std::size_t i = 0; i < refined.provenance.size(); ++i) {
    const auto &record = refined.provenance[i];
    if (record.state == refit_state::refit) {
      ++refits;
      continue;
    }
    clamped += record.state == refit_state::clamped;
    INFO("vertex " << i << " state " << int(record.state));
    REQUIRE(record.moved == 0.f);
    for (int d = 0; d < 3; ++d)
      REQUIRE(refined.mesh.points()[i][d] == plain.mesh.points()[i][d]);
  }
  REQUIRE(refits > 0);
  // the verdict whose vertex the pass used to overwrite with the pool centroid
  REQUIRE(clamped > 0);
}

// Edge degree states nothing about area or orientation: the fan completion can
// emit a sliver and a refit can fold a quad that straddles a crease. Both are
// bounded here, and away from a crease the triangle normal never opposes the
// field's own gradient.
//
// The octahedron's bound is one, not zero, and that is the honest number: a
// zero-area triangle here names three distinct vertices whose positions a refit
// brought together, and dropping it would take two edges out of a surface that
// is manifold by construction — trading a sliver for a hole. The count is
// pinned so the day it grows is a red test.
TEST_CASE("the triangles carry area and face outward",
          "[volume][sharpness]") {
  const int n = 48;
  for (const bool refine : {true, false}) {
    {
      const sharp_field::box shape;
      const auto out = sharp_extract(shape, n, refine);
      const auto r =
          sharp_inspect_triangles(shape, out.mesh.polygons(), out.spacing);
      INFO("box refine = " << refine << " degenerate " << r.degenerate
                           << " inverted_far " << r.inverted_far << " of "
                           << r.n_far);
      REQUIRE(r.n_far > 1000u);
      REQUIRE(r.inverted_far == 0u);
      REQUIRE(r.degenerate == 0u);
    }
    {
      const sharp_field::octahedron shape;
      const auto out = sharp_extract(shape, n, refine);
      const auto r =
          sharp_inspect_triangles(shape, out.mesh.polygons(), out.spacing);
      INFO("octahedron refine = " << refine << " degenerate " << r.degenerate
                                  << " inverted_far " << r.inverted_far
                                  << " of " << r.n_far);
      REQUIRE(r.n_far > 500u);
      REQUIRE(r.inverted_far == 0u);
      REQUIRE(r.degenerate <= 1u);
    }
  }
}

// The request is in the mould of record_polygons: a build that did not ask
// states nothing, and one that did states one record per patch vertex.
TEST_CASE("the refit provenance is a request", "[volume][sharpness]") {
  const auto vol = tf::test::sampled_volume(sharp_field::box{}, 32);
  tf::isosurface_config config;
  config.method = tf::isosurface_method::dual_contouring;
  {
    tf::volume_detail::dual_contouring<int, float> dc;
    const auto mesh = dc.build(vol.volume(), 0.f, config);
    REQUIRE(mesh.polygons().size() > 0);
    REQUIRE(dc.refit_provenance().size() == 0);
  }
  {
    tf::volume_detail::dual_contouring<int, float> dc;
    dc.record_refit_provenance(true);
    const auto mesh = dc.build(vol.volume(), 0.f, config);
    const auto &prov = dc.refit_provenance();
    REQUIRE(prov.size() > 0);
    REQUIRE(prov.size() <= std::size_t(mesh.points().size()));
    for (std::size_t i = 0; i < prov.size(); ++i)
      if (prov[i].state == refit_state::unrefined)
        REQUIRE(prov[i].moved == 0.f);
  }
  {
    // the refinement it accounts for is the one the config asked for
    tf::volume_detail::dual_contouring<int, float> dc;
    dc.record_refit_provenance(true);
    config.refine = false;
    const auto mesh = dc.build(vol.volume(), 0.f, config);
    REQUIRE(mesh.polygons().size() > 0);
    REQUIRE(dc.refit_provenance().size() == 0);
  }
}
