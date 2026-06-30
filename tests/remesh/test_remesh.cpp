/**
 * @file test_remesh.cpp
 * @brief Tests for decimation and isotropic remeshing
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/trueform.hpp>
#include <trueform/remesh/protect_vertices.hpp>

#include <cmath>
#include <vector>

TEST_CASE("decimated (box to 50%)", "[remesh][decimate]") {
  auto box = tf::make_box_mesh(2.0f, 3.0f, 4.0f);
  auto orig_faces = box.faces().size();

  auto [dec, he] = tf::decimated(box.polygons(), 0.5f);
  REQUIRE(dec.faces().size() > 0);
  REQUIRE(dec.faces().size() <= orig_faces);
  REQUIRE(dec.points().size() > 0);
  REQUIRE(dec.points().size() <= box.points().size());
}

TEST_CASE("decimated (sphere -preserves rough shape)", "[remesh][decimate]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 20, 20);
  auto orig_faces = sphere.faces().size();

  auto [dec, he] = tf::decimated(sphere.polygons(), 0.3f);
  REQUIRE(dec.faces().size() < orig_faces);
  REQUIRE(dec.faces().size() > 0);

  auto orig_vol = std::abs(tf::signed_volume(sphere.polygons()));
  auto dec_vol = std::abs(tf::signed_volume(dec.polygons()));
  auto ratio = dec_vol / orig_vol;
  REQUIRE(ratio > 0.5f);
  REQUIRE(ratio < 1.5f);
}

TEST_CASE("decimated with options", "[remesh][decimate]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 15, 15);

  tf::decimate_config<float> config(
      /*min_quality=*/0.3f,
      /*preserve_boundary=*/false,
      /*parallel=*/true,
      /*feature_angle=*/tf::rad<float>(-1),
      /*feature_weight=*/100.0f,
      /*stabilizer=*/1e-3);
  auto [dec, he] = tf::decimated(sphere.polygons(), 0.5f, config);
  REQUIRE(dec.faces().size() > 0);
  REQUIRE(dec.faces().size() <= sphere.faces().size());
}

TEST_CASE("simplified (error budget collapses flat faces)", "[remesh][simplify]") {
  auto box = tf::make_box_mesh(2.0f, 3.0f, 4.0f, 4, 4, 4);
  auto orig = box.faces().size();
  auto [s, he] = tf::simplified(box.polygons());
  REQUIRE(s.faces().size() > 0);
  REQUIRE(s.faces().size() < orig); // flat box faces collapse for ~0 error
}

TEST_CASE("simplified (larger budget is coarser)", "[remesh][simplify]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 30, 30);
  auto fine = tf::simplified(sphere.polygons(), tf::simplify_config<float>{0.002f}).first;
  auto coarse = tf::simplified(sphere.polygons(), tf::simplify_config<float>{0.02f}).first;
  REQUIRE(coarse.faces().size() > 0);
  REQUIRE(coarse.faces().size() <= fine.faces().size());
}

TEST_CASE("simplified (centering: large coords do not stall)",
          "[remesh][simplify]") {
  // The bbox-min centering must keep the quadric error meaningful at large
  // (e.g. UTM) coordinates; without it a translated mesh barely simplifies.
  auto sphere = tf::make_sphere_mesh(1.0f, 30, 30);
  auto orig = sphere.faces().size();
  tf::simplify_config<float> cfg{0.01f};
  auto base = tf::simplified(sphere.polygons(), cfg).first;
  REQUIRE(base.faces().size() < orig);

  auto moved = sphere;
  for (std::size_t i = 0; i < moved.points().size(); ++i)
    for (int d = 0; d < 3; ++d)
      moved.points()[i][d] += 500000.0f;
  auto moveds = tf::simplified(moved.polygons(), cfg).first;
  REQUIRE(moveds.faces().size() < orig);                    // simplifies at scale
  REQUIRE(moveds.faces().size() < base.faces().size() * 2); // not stalled
}

namespace {
// Worst minimum interior angle over all faces (radians).
template <typename HE, typename Pts>
float mesh_min_angle(const HE &he, const Pts &points) {
  auto faces = tf::make_faces_buffer(he);
  float worst = 4.0f;
  for (decltype(faces.size()) f = 0; f < faces.size(); ++f) {
    auto t = faces[f];
    auto A = points[t[0]], B = points[t[1]], C = points[t[2]];
    auto L = [](auto p, auto q) {
      float dx = p[0]-q[0], dy = p[1]-q[1], dz = p[2]-q[2];
      return std::sqrt(dx*dx + dy*dy + dz*dz);
    };
    float a = L(B,C), b = L(A,C), c = L(A,B);
    auto ang = [&](float o, float s1, float s2) {
      if (s1 <= 0 || s2 <= 0) return 0.0f;
      float co = (s1*s1 + s2*s2 - o*o) / (2*s1*s2);
      co = std::max(-1.0f, std::min(1.0f, co));
      return std::acos(co);
    };
    worst = std::min({worst, ang(a,b,c), ang(b,a,c), ang(c,a,b)});
  }
  return worst;
}
} // namespace

TEST_CASE("improve_triangulation (fixed count)", "[remesh][improve]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 20, 20);
  tf::half_edges<int> he(sphere.polygons());
  auto points = sphere.points_buffer();
  auto nf = he.number_of_faces();
  auto nv = points.size();
  tf::improve_triangulation(he, points.points()); // default: valence, 3 rounds
  REQUIRE(he.number_of_faces() == nf); // flips + relax never change the count
  REQUIRE(points.size() == nv);
}

TEST_CASE("improve_triangulation (min_angle objective, no folds)",
          "[remesh][improve]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 25, 25);
  tf::half_edges<int> he(sphere.polygons());
  auto points = sphere.points_buffer();
  tf::improve_config<float> cfg;
  cfg.flip = tf::flip_objective::min_angle;
  cfg.check_normals = true;
  tf::improve_triangulation(he, points.points(), cfg);
  // the fold guard must keep every triangle non-degenerate / non-inverted
  REQUIRE(mesh_min_angle(he, points.points()) > 0.01f); // > ~0.6 degrees
}

TEST_CASE("improve_triangulation (reused workspace)", "[remesh][improve]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 15, 15);
  tf::half_edges<int> he(sphere.polygons());
  auto points = sphere.points_buffer();
  auto nf = he.number_of_faces();
  tf::points_buffer<double, 3> old_pos;
  tf::improve_config<float> cfg{1, 2, 0.5f, true, tf::flip_objective::valence};
  tf::improve_triangulation(he, points.points(), old_pos, cfg);
  tf::improve_triangulation(he, points.points(), old_pos, cfg); // reuse buffer
  REQUIRE(he.number_of_faces() == nf);
}

TEST_CASE("isotropicRemeshed (box -basic)", "[remesh][isotropic]") {
  auto box = tf::make_box_mesh(2.0f, 3.0f, 4.0f, 3, 3, 3);
  auto mel = tf::mean_edge_length(box.polygons());

  auto [rem, he] = tf::isotropic_remeshed(box.polygons(), mel * 2.0f);
  REQUIRE(rem.faces().size() > 0);
  REQUIRE(rem.points().size() > 0);
}

TEST_CASE("isotropicRemeshed (sphere -edge lengths converge)",
          "[remesh][isotropic]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 10, 10);
  auto target = tf::mean_edge_length(sphere.polygons());

  tf::isotropic_remesh_config<float> config(target, /*iterations=*/5);
  auto [rem, he] = tf::isotropic_remeshed(sphere.polygons(), config);
  REQUIRE(rem.faces().size() > 0);

  auto min_el = tf::min_edge_length(rem.polygons());
  auto max_el = tf::max_edge_length(rem.polygons());
  auto ratio = max_el / min_el;
  REQUIRE(ratio < 10.0f);
}

TEST_CASE("isotropicRemeshed with quadric", "[remesh][isotropic]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 10, 10);
  auto target = tf::mean_edge_length(sphere.polygons());

  tf::isotropic_remesh_config<float> config(target, /*iterations=*/3);
  config.use_quadric = true;
  auto [rem, he] = tf::isotropic_remeshed(sphere.polygons(), config);
  REQUIRE(rem.faces().size() > 0);
}

TEST_CASE("pipeline: decimate then isotropic remesh",
          "[remesh][pipeline]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 20, 20);
  auto orig_faces = sphere.faces().size();

  auto [dec, he1] = tf::decimated(sphere.polygons(), 0.3f);
  REQUIRE(dec.faces().size() < orig_faces);

  auto mel = tf::mean_edge_length(dec.polygons());
  tf::isotropic_remesh_config<float> config(mel, /*iterations=*/3);
  config.use_quadric = true;
  auto [rem, he2] = tf::isotropic_remeshed(dec.polygons(), config);
  REQUIRE(rem.faces().size() > 0);

  auto orig_vol = std::abs(tf::signed_volume(sphere.polygons()));
  auto rem_vol = std::abs(tf::signed_volume(rem.polygons()));
  auto ratio = rem_vol / orig_vol;
  REQUIRE(ratio > 0.3f);
  REQUIRE(ratio < 2.0f);
}

TEST_CASE("isotropicRemeshed with preserveBoundary",
          "[remesh][isotropic]") {
  auto plane = tf::make_plane_mesh(10.0f, 10.0f, 5, 5);
  auto mel = tf::mean_edge_length(plane.polygons());

  tf::isotropic_remesh_config<float> config(mel, /*iterations=*/2);
  config.preserve_boundary = true;
  auto [rem, he] = tf::isotropic_remeshed(plane.polygons(), config);
  REQUIRE(rem.faces().size() > 0);
}

namespace {
// Forward+reverse round-trip of an original->new vertex index map: every kept
// vertex inverts cleanly, and every surviving original maps back to itself.
template <typename Map>
void require_vertex_map_consistent(const Map &vmap, std::size_t n_orig,
                                   std::size_t n_final) {
  const auto &f = vmap.f();
  const auto &kept = vmap.kept_ids();
  REQUIRE(f.size() == n_orig);
  REQUIRE(kept.size() == n_final);
  for (std::size_t j = 0; j < n_final; ++j)
    REQUIRE(std::size_t(f[kept[j]]) == j);
  for (std::size_t v = 0; v < n_orig; ++v) {
    auto j = f[v];
    if (std::size_t(j) < n_final)
      REQUIRE(std::size_t(kept[j]) == v);
  }
}
} // namespace

namespace {
// Count the protected vertices in an output protection mask.
template <typename Mask> std::size_t count_true(const Mask &m) {
  std::size_t n = 0;
  for (std::size_t i = 0; i < m.size(); ++i)
    n += m[i] ? 1 : 0;
  return n;
}
} // namespace

TEST_CASE("simplified protect_vertices (returns the new protection mask)",
          "[remesh][simplify][protect]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 30, 30);
  const auto nv = sphere.points().size();
  std::vector<int> mask(nv, 0);
  const std::vector<int> pinned = {0, 5, 17, 100, 250};
  for (int v : pinned)
    mask[v] = 1;

  // protect alone returns (mesh, he, protection_mask) -- NO index map.
  auto [s, he, prot] = tf::simplified(
      sphere.polygons(), tf::simplify_config<float>{0.02f},
      tf::protect_vertices(mask));

  REQUIRE(s.faces().size() < sphere.faces().size());
  REQUIRE(prot.size() == s.points().size());      // per-output-vertex
  REQUIRE(count_true(prot) == pinned.size());      // exactly the pinned survive
}

TEST_CASE("simplified protect_vertices + index map (survive, position, mask)",
          "[remesh][simplify][protect][indexmap]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 30, 30);
  const auto nv = sphere.points().size();
  std::vector<int> mask(nv, 0);
  const std::vector<int> pinned = {0, 5, 17, 100, 250};
  for (int v : pinned)
    mask[v] = 1;

  // protect + return_index_map -> (mesh, he, protection_mask, vertex_map),
  // in parameter order (protect before return_index_map).
  auto [s, he, prot, vmap] = tf::simplified(
      sphere.polygons(), tf::simplify_config<float>{0.02f},
      tf::protect_vertices(mask), tf::return_index_map);

  require_vertex_map_consistent(vmap, nv, s.points().size());
  REQUIRE(prot.size() == s.points().size());
  REQUIRE(count_true(prot) == pinned.size());

  for (int v : pinned) {
    auto j = vmap.f()[v];
    REQUIRE(std::size_t(j) < s.points().size()); // never collapsed away
    REQUIRE(prot[j]);                            // marked in the output mask
    float d2 = 0;                                // position kept (centering eps)
    for (int k = 0; k < 3; ++k) {
      float e = s.points()[j][k] - sphere.points()[v][k];
      d2 += e * e;
    }
    REQUIRE(d2 < 1e-8f);
  }
}

TEST_CASE("simplified protect_vertices + map (multi-iteration composes)",
          "[remesh][simplify][protect][indexmap]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 40, 40);
  const auto nv = sphere.points().size();
  std::vector<int> mask(nv, 0);
  const std::vector<int> pinned = {0, 3, 11, 400};
  for (int v : pinned)
    mask[v] = 1;

  tf::simplify_config<float> cfg{0.01f};
  cfg.iterations = 4; // > 1: folds 4 passes through compose_index_maps

  auto [s, he, prot, vmap] = tf::simplified(
      sphere.polygons(), cfg, tf::protect_vertices(mask), tf::return_index_map);

  REQUIRE(s.faces().size() < sphere.faces().size());
  require_vertex_map_consistent(vmap, nv, s.points().size());
  REQUIRE(count_true(prot) == pinned.size());
  for (int v : pinned) {
    auto j = vmap.f()[v];
    REQUIRE(std::size_t(j) < s.points().size()); // survive every pass
    REQUIRE(prot[j]);
  }
}

TEST_CASE("simplified empty protect mask (all-false mask, matches plain)",
          "[remesh][simplify][protect]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 25, 25);
  std::vector<int> empty;
  tf::simplify_config<float> cfg{0.01f};

  auto [s, he, prot] =
      tf::simplified(sphere.polygons(), cfg, tf::protect_vertices(empty));
  auto plain = tf::simplified(sphere.polygons(), cfg).first;

  REQUIRE(prot.size() == 0); // empty input mask -> empty protection mask
  // Pinning nothing matches plain up to parallel-collapse nondeterminism.
  double a = double(s.faces().size()), b = double(plain.faces().size());
  REQUIRE(std::abs(a - b) <= 0.1 * b);
}

TEST_CASE("simplified return_index_map (no protection)",
          "[remesh][simplify][indexmap]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 25, 25);
  auto [s, he, vmap] = tf::simplified(
      sphere.polygons(), tf::simplify_config<float>{0.01f}, tf::return_index_map);
  require_vertex_map_consistent(vmap, sphere.points().size(), s.points().size());
}

TEST_CASE("simplified preserve_regions + protect (mask + labels)",
          "[remesh][simplify][protect][regions]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 30, 30);
  const auto nv = sphere.points().size();
  const auto nf = sphere.faces().size();
  std::vector<int> labels(nf, 0);
  for (std::size_t i = 0; i < nf; ++i)
    labels[i] = (i < nf / 2) ? 0 : 1; // two regions
  std::vector<int> mask(nv, 0);
  mask[0] = 1;
  mask[10] = 1;

  // regions + protect -> (mesh, he, face_labels, protection_mask).
  auto [s, he, lab, prot] = tf::simplified(
      sphere.polygons(), tf::simplify_config<float>{0.02f},
      tf::preserve_regions(labels), tf::protect_vertices(mask));

  REQUIRE(lab.size() == s.faces().size());    // one label per output face
  REQUIRE(prot.size() == s.points().size());  // one flag per output vertex
  REQUIRE(count_true(prot) == 2);
}

TEST_CASE("simplified preserve_regions + protect + index map (5-tuple)",
          "[remesh][simplify][protect][regions][indexmap]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 30, 30);
  const auto nv = sphere.points().size();
  const auto nf = sphere.faces().size();
  std::vector<int> labels(nf, 0);
  for (std::size_t i = 0; i < nf; ++i)
    labels[i] = (i < nf / 2) ? 0 : 1;
  std::vector<int> mask(nv, 0);
  mask[0] = 1;
  mask[10] = 1;

  // -> (mesh, he, face_labels, protection_mask, vertex_map) in param order.
  auto [s, he, lab, prot, vmap] = tf::simplified(
      sphere.polygons(), tf::simplify_config<float>{0.02f},
      tf::preserve_regions(labels), tf::protect_vertices(mask),
      tf::return_index_map);

  REQUIRE(lab.size() == s.faces().size());
  REQUIRE(prot.size() == s.points().size());
  require_vertex_map_consistent(vmap, nv, s.points().size());
  REQUIRE(prot[vmap.f()[0]]);
  REQUIRE(prot[vmap.f()[10]]);
}

// --- decimate: pure collapse, so face AND vertex maps are exact ------------

TEST_CASE("decimated return_index_map (face + vertex maps)",
          "[remesh][decimate][indexmap]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 20, 20);
  auto [d, he, fim, vim] =
      tf::decimated(sphere.polygons(), 0.4f, tf::return_index_map);
  require_vertex_map_consistent(vim, sphere.points().size(), d.points().size());
  REQUIRE(fim.kept_ids().size() == d.faces().size()); // a kept id per out face
}

TEST_CASE("decimated protect_vertices (returns mask)",
          "[remesh][decimate][protect]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 25, 25);
  auto nv = sphere.points().size();
  std::vector<int> mask(nv, 0);
  mask[0] = 1;
  mask[9] = 1;
  mask[40] = 1;
  auto [d, he, prot] =
      tf::decimated(sphere.polygons(), 0.3f, tf::protect_vertices(mask));
  REQUIRE(prot.size() == d.points().size());
  REQUIRE(count_true(prot) == 3);
}

TEST_CASE("decimated protect_vertices + maps (survival via vertex map)",
          "[remesh][decimate][protect][indexmap]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 30, 30);
  auto nv = sphere.points().size();
  std::vector<int> mask(nv, 0);
  const std::vector<int> pinned = {0, 7, 22, 150};
  for (int v : pinned)
    mask[v] = 1;
  // protect + map -> (mesh, he, protection_mask, face_map, vertex_map).
  auto [d, he, prot, fim, vim] = tf::decimated(
      sphere.polygons(), 0.3f, tf::protect_vertices(mask), tf::return_index_map);
  REQUIRE(d.points().size() < nv);
  REQUIRE(prot.size() == d.points().size());
  REQUIRE(count_true(prot) == pinned.size());
  require_vertex_map_consistent(vim, nv, d.points().size());
  for (int v : pinned) {
    auto j = vim.f()[v];
    REQUIRE(std::size_t(j) < d.points().size()); // never collapsed away
    REQUIRE(prot[j]);                            // marked in output mask
  }
}

TEST_CASE("decimated empty protect mask (empty mask out)",
          "[remesh][decimate][protect]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 20, 20);
  std::vector<int> empty;
  auto [d, he, prot] =
      tf::decimated(sphere.polygons(), 0.5f, tf::protect_vertices(empty));
  REQUIRE(prot.size() == 0); // empty input mask -> empty protection mask
}

TEST_CASE("decimated preserve_regions + protect (labels + mask)",
          "[remesh][decimate][protect][regions]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 25, 25);
  auto nv = sphere.points().size();
  auto nf = sphere.faces().size();
  std::vector<int> labels(nf, 0);
  for (std::size_t i = 0; i < nf; ++i)
    labels[i] = (i < nf / 2) ? 0 : 1;
  std::vector<int> mask(nv, 0);
  mask[0] = 1;
  mask[5] = 1;
  // param order: regions before protect -> (mesh, he, face_labels, mask).
  auto [d, he, lab, prot] =
      tf::decimated(sphere.polygons(), 0.5f, tf::preserve_regions(labels),
                    tf::protect_vertices(mask));
  REQUIRE(lab.size() == d.faces().size());
  REQUIRE(prot.size() == d.points().size());
  REQUIRE(count_true(prot) == 2);
}

// --- collapse_short_edges: pure collapse, face + vertex maps exact ---------

TEST_CASE("collapsed_short_edges protect_vertices + maps",
          "[remesh][collapse][protect][indexmap]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 30, 30);
  auto nv = sphere.points().size();
  auto mel = tf::mean_edge_length(sphere.polygons());
  std::vector<int> mask(nv, 0);
  const std::vector<int> pinned = {0, 7, 22, 150};
  for (int v : pinned)
    mask[v] = 1;
  // protect + map -> (mesh, he, protection_mask, face_map, vertex_map).
  auto [c, he, prot, fim, vim] =
      tf::collapsed_short_edges(sphere.polygons(), 0.6f * mel,
                                tf::protect_vertices(mask), tf::return_index_map);
  REQUIRE(prot.size() == c.points().size());
  REQUIRE(count_true(prot) == pinned.size());
  require_vertex_map_consistent(vim, nv, c.points().size());
  REQUIRE(fim.kept_ids().size() == c.faces().size());
  for (int v : pinned) {
    auto j = vim.f()[v];
    REQUIRE(std::size_t(j) < c.points().size());
    REQUIRE(prot[j]);
  }
}

TEST_CASE("collapsed_short_edges empty protect mask (empty mask out)",
          "[remesh][collapse][protect]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 20, 20);
  auto mel = tf::mean_edge_length(sphere.polygons());
  std::vector<int> empty;
  auto [c, he, prot] = tf::collapsed_short_edges(sphere.polygons(), 0.6f * mel,
                                                 tf::protect_vertices(empty));
  REQUIRE(prot.size() == 0);
}

TEST_CASE("collapsed_short_edges preserve_regions + protect (labels + mask)",
          "[remesh][collapse][protect][regions]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 25, 25);
  auto nv = sphere.points().size();
  auto nf = sphere.faces().size();
  auto mel = tf::mean_edge_length(sphere.polygons());
  std::vector<int> labels(nf, 0);
  for (std::size_t i = 0; i < nf; ++i)
    labels[i] = (i < nf / 2) ? 0 : 1;
  std::vector<int> mask(nv, 0);
  mask[0] = 1;
  mask[5] = 1;
  auto [c, he, lab, prot] = tf::collapsed_short_edges(
      sphere.polygons(), 0.6f * mel, tf::preserve_regions(labels),
      tf::protect_vertices(mask));
  REQUIRE(lab.size() == c.faces().size());
  REQUIRE(prot.size() == c.points().size());
  REQUIRE(count_true(prot) == 2);
}

// --- isotropic remesh: splits add vertices; vertex map is forward-derived ---

TEST_CASE("isotropic_remeshed protect + map (splits: kept_ids original or none)",
          "[remesh][isotropic][protect][indexmap]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 20, 20);
  auto nv = sphere.points().size();
  auto mel = tf::mean_edge_length(sphere.polygons());
  std::vector<int> mask(nv, 0);
  const std::vector<int> pinned = {0, 5, 30, 100};
  for (int v : pinned)
    mask[v] = 1;
  tf::isotropic_remesh_config<float> cfg(mel * 0.5f, 3); // small target -> splits

  auto [r, he, prot, vmap] = tf::isotropic_remeshed(
      sphere.polygons(), cfg, tf::protect_vertices(mask), tf::return_index_map);

  auto nf = r.points().size();
  REQUIRE(prot.size() == nf);
  REQUIRE(count_true(prot) == pinned.size());

  const auto &f = vmap.f();
  const auto &kept = vmap.kept_ids();
  REQUIRE(f.size() == nv);
  REQUIRE(kept.size() == nf);

  // surviving originals round-trip
  for (std::size_t v = 0; v < nv; ++v) {
    auto j = f[v];
    if (std::size_t(j) < nf)
      REQUIRE(std::size_t(kept[j]) == v);
  }
  // every output vertex is either an original survivor or a split vertex whose
  // kept_ids entry is the none sentinel (== original vertex count)
  std::size_t split_count = 0;
  for (std::size_t j = 0; j < nf; ++j) {
    auto o = kept[j];
    if (std::size_t(o) == nv) {
      ++split_count; // split-created: no original preimage
    } else {
      REQUIRE(std::size_t(o) < nv);
      REQUIRE(std::size_t(f[o]) == j);
    }
  }
  REQUIRE(split_count > 0); // splits really happened
  // pinned survived every split/collapse pass and are marked
  for (int v : pinned) {
    auto j = f[v];
    REQUIRE(std::size_t(j) < nf);
    REQUIRE(prot[j]);
  }
}

TEST_CASE("isotropic_remeshed empty protect mask (empty mask out)",
          "[remesh][isotropic][protect]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 12, 12);
  auto mel = tf::mean_edge_length(sphere.polygons());
  std::vector<int> empty;
  auto [r, he, prot] = tf::isotropic_remeshed(sphere.polygons(), mel,
                                              tf::protect_vertices(empty));
  REQUIRE(prot.size() == 0);
}

TEST_CASE("isotropic_remeshed preserve_regions + protect (labels + mask)",
          "[remesh][isotropic][protect][regions]") {
  auto sphere = tf::make_sphere_mesh(1.0f, 15, 15);
  auto nv = sphere.points().size();
  auto nf = sphere.faces().size();
  auto mel = tf::mean_edge_length(sphere.polygons());
  std::vector<int> labels(nf, 0);
  for (std::size_t i = 0; i < nf; ++i)
    labels[i] = (i < nf / 2) ? 0 : 1;
  std::vector<int> mask(nv, 0);
  mask[0] = 1;
  mask[3] = 1;
  auto [r, he, lab, prot] = tf::isotropic_remeshed(
      sphere.polygons(), mel, tf::preserve_regions(labels),
      tf::protect_vertices(mask));
  REQUIRE(lab.size() == r.faces().size());
  REQUIRE(prot.size() == r.points().size());
  REQUIRE(count_true(prot) == 2);
}

// --- exhaustive overload coverage: every axis combo per function -----------

namespace {
// Survivor-only vertex-map check (valid even when splits add new vertices, as
// in isotropic): every surviving original round-trips; split verts are skipped.
template <typename Map>
void require_vmap_survivors(const Map &vmap, std::size_t n_orig,
                           std::size_t n_final) {
  const auto &f = vmap.f();
  const auto &kept = vmap.kept_ids();
  REQUIRE(f.size() == n_orig);
  REQUIRE(kept.size() == n_final);
  for (std::size_t v = 0; v < n_orig; ++v) {
    auto j = f[v];
    if (std::size_t(j) < n_final)
      REQUIRE(std::size_t(kept[j]) == v);
  }
}
} // namespace

TEST_CASE("simplified: every overload combo", "[remesh][simplify][coverage]") {
  auto s = tf::make_sphere_mesh(1.0f, 25, 25);
  auto nv = s.points().size(), nf = s.faces().size();
  tf::simplify_config<float> c{0.01f};
  std::vector<int> mask(nv, 0); mask[0] = 1; mask[5] = 1;
  std::vector<int> lab(nf, 0);
  for (std::size_t i = 0; i < nf; ++i) lab[i] = (i < nf / 2) ? 0 : 1;
  auto P = [&] { return tf::protect_vertices(mask); };
  auto R = [&] { return tf::preserve_regions(lab); };

  { auto [m, h] = tf::simplified(s.polygons(), c); REQUIRE(m.faces().size() > 0); }
  { auto [m, h, vm] = tf::simplified(s.polygons(), c, tf::return_index_map);
    require_vertex_map_consistent(vm, nv, m.points().size()); }
  { auto [m, h, pr] = tf::simplified(s.polygons(), c, P());
    REQUIRE(pr.size() == m.points().size()); REQUIRE(count_true(pr) == 2); }
  { auto [m, h, pr, vm] = tf::simplified(s.polygons(), c, P(), tf::return_index_map);
    REQUIRE(count_true(pr) == 2); require_vertex_map_consistent(vm, nv, m.points().size()); }
  { auto [m, h, l] = tf::simplified(s.polygons(), c, R()); REQUIRE(l.size() == m.faces().size()); }
  { auto [m, h, l, vm] = tf::simplified(s.polygons(), c, R(), tf::return_index_map);
    REQUIRE(l.size() == m.faces().size()); require_vertex_map_consistent(vm, nv, m.points().size()); }
  { auto [m, h, l, pr] = tf::simplified(s.polygons(), c, R(), P());
    REQUIRE(l.size() == m.faces().size()); REQUIRE(pr.size() == m.points().size()); }
  { auto [m, h, l, pr, vm] = tf::simplified(s.polygons(), c, R(), P(), tf::return_index_map);
    REQUIRE(l.size() == m.faces().size()); REQUIRE(count_true(pr) == 2);
    require_vertex_map_consistent(vm, nv, m.points().size()); }
  // default-config variants (compile + run)
  { auto [m, h] = tf::simplified(s.polygons()); REQUIRE(m.faces().size() > 0); }
  { auto [m, h, vm] = tf::simplified(s.polygons(), tf::return_index_map); (void)vm; }
  { auto [m, h, pr] = tf::simplified(s.polygons(), P()); (void)pr; }
  { auto [m, h, l, pr, vm] = tf::simplified(s.polygons(), R(), P(), tf::return_index_map);
    (void)l; (void)pr; (void)vm; }
}

TEST_CASE("decimated: every overload combo", "[remesh][decimate][coverage]") {
  auto s = tf::make_sphere_mesh(1.0f, 25, 25);
  auto nv = s.points().size(), nf = s.faces().size();
  tf::decimate_config<float> c{};
  std::vector<int> mask(nv, 0); mask[0] = 1; mask[5] = 1;
  std::vector<int> lab(nf, 0);
  for (std::size_t i = 0; i < nf; ++i) lab[i] = (i < nf / 2) ? 0 : 1;
  auto P = [&] { return tf::protect_vertices(mask); };
  auto R = [&] { return tf::preserve_regions(lab); };

  { auto [m, h] = tf::decimated(s.polygons(), 0.4f, c); REQUIRE(m.faces().size() > 0); }
  { auto [m, h, fm, vm] = tf::decimated(s.polygons(), 0.4f, c, tf::return_index_map);
    require_vertex_map_consistent(vm, nv, m.points().size());
    REQUIRE(fm.kept_ids().size() == m.faces().size()); }
  { auto [m, h, pr] = tf::decimated(s.polygons(), 0.4f, c, P());
    REQUIRE(count_true(pr) == 2); }
  { auto [m, h, pr, fm, vm] = tf::decimated(s.polygons(), 0.4f, c, P(), tf::return_index_map);
    REQUIRE(count_true(pr) == 2); require_vertex_map_consistent(vm, nv, m.points().size());
    REQUIRE(fm.kept_ids().size() == m.faces().size()); }
  { auto [m, h, l] = tf::decimated(s.polygons(), 0.4f, c, R()); REQUIRE(l.size() == m.faces().size()); }
  { auto [m, h, l, fm, vm] = tf::decimated(s.polygons(), 0.4f, c, R(), tf::return_index_map);
    REQUIRE(l.size() == m.faces().size()); require_vertex_map_consistent(vm, nv, m.points().size()); }
  { auto [m, h, l, pr] = tf::decimated(s.polygons(), 0.4f, c, R(), P());
    REQUIRE(l.size() == m.faces().size()); REQUIRE(pr.size() == m.points().size()); }
  { auto [m, h, l, pr, fm, vm] = tf::decimated(s.polygons(), 0.4f, c, R(), P(), tf::return_index_map);
    REQUIRE(l.size() == m.faces().size()); REQUIRE(count_true(pr) == 2);
    require_vertex_map_consistent(vm, nv, m.points().size());
    REQUIRE(fm.kept_ids().size() == m.faces().size()); }
  // default-config variants
  { auto [m, h] = tf::decimated(s.polygons(), 0.4f); REQUIRE(m.faces().size() > 0); }
  { auto [m, h, fm, vm] = tf::decimated(s.polygons(), 0.4f, tf::return_index_map); (void)fm; (void)vm; }
  { auto [m, h, l, pr, fm, vm] = tf::decimated(s.polygons(), 0.4f, R(), P(), tf::return_index_map);
    (void)l; (void)pr; (void)fm; (void)vm; }
}

TEST_CASE("collapsed_short_edges: every overload combo",
          "[remesh][collapse][coverage]") {
  auto s = tf::make_sphere_mesh(1.0f, 22, 22);
  auto nv = s.points().size(), nf = s.faces().size();
  auto mel = tf::mean_edge_length(s.polygons());
  float ml = 0.6f * mel;
  tf::length_collapse_config<float> c{};
  std::vector<int> mask(nv, 0); mask[0] = 1; mask[5] = 1;
  std::vector<int> lab(nf, 0);
  for (std::size_t i = 0; i < nf; ++i) lab[i] = (i < nf / 2) ? 0 : 1;
  auto P = [&] { return tf::protect_vertices(mask); };
  auto R = [&] { return tf::preserve_regions(lab); };

  { auto [m, h] = tf::collapsed_short_edges(s.polygons(), ml, c); REQUIRE(m.faces().size() > 0); }
  { auto [m, h, fm, vm] = tf::collapsed_short_edges(s.polygons(), ml, c, tf::return_index_map);
    require_vertex_map_consistent(vm, nv, m.points().size());
    REQUIRE(fm.kept_ids().size() == m.faces().size()); }
  { auto [m, h, pr] = tf::collapsed_short_edges(s.polygons(), ml, c, P());
    REQUIRE(count_true(pr) == 2); }
  { auto [m, h, pr, fm, vm] = tf::collapsed_short_edges(s.polygons(), ml, c, P(), tf::return_index_map);
    REQUIRE(count_true(pr) == 2); require_vertex_map_consistent(vm, nv, m.points().size()); }
  { auto [m, h, l] = tf::collapsed_short_edges(s.polygons(), ml, c, R()); REQUIRE(l.size() == m.faces().size()); }
  { auto [m, h, l, fm, vm] = tf::collapsed_short_edges(s.polygons(), ml, c, R(), tf::return_index_map);
    REQUIRE(l.size() == m.faces().size()); require_vertex_map_consistent(vm, nv, m.points().size()); }
  { auto [m, h, l, pr] = tf::collapsed_short_edges(s.polygons(), ml, c, R(), P());
    REQUIRE(l.size() == m.faces().size()); REQUIRE(pr.size() == m.points().size()); }
  { auto [m, h, l, pr, fm, vm] = tf::collapsed_short_edges(s.polygons(), ml, c, R(), P(), tf::return_index_map);
    REQUIRE(l.size() == m.faces().size()); REQUIRE(count_true(pr) == 2);
    require_vertex_map_consistent(vm, nv, m.points().size()); }
  // default-config variants
  { auto [m, h] = tf::collapsed_short_edges(s.polygons(), ml); REQUIRE(m.faces().size() > 0); }
  { auto [m, h, fm, vm] = tf::collapsed_short_edges(s.polygons(), ml, tf::return_index_map); (void)fm; (void)vm; }
  { auto [m, h, l, pr, fm, vm] = tf::collapsed_short_edges(s.polygons(), ml, R(), P(), tf::return_index_map);
    (void)l; (void)pr; (void)fm; (void)vm; }
}

TEST_CASE("isotropic_remeshed: every overload combo",
          "[remesh][isotropic][coverage]") {
  auto s = tf::make_sphere_mesh(1.0f, 16, 16);
  auto nv = s.points().size(), nf = s.faces().size();
  auto mel = tf::mean_edge_length(s.polygons());
  tf::isotropic_remesh_config<float> c(mel, 2);
  std::vector<int> mask(nv, 0); mask[0] = 1; mask[3] = 1;
  std::vector<int> lab(nf, 0);
  for (std::size_t i = 0; i < nf; ++i) lab[i] = (i < nf / 2) ? 0 : 1;
  auto P = [&] { return tf::protect_vertices(mask); };
  auto R = [&] { return tf::preserve_regions(lab); };

  { auto [m, h] = tf::isotropic_remeshed(s.polygons(), c); REQUIRE(m.faces().size() > 0); }
  { auto [m, h, vm] = tf::isotropic_remeshed(s.polygons(), c, tf::return_index_map);
    require_vmap_survivors(vm, nv, m.points().size()); }
  { auto [m, h, pr] = tf::isotropic_remeshed(s.polygons(), c, P());
    REQUIRE(pr.size() == m.points().size()); REQUIRE(count_true(pr) == 2); }
  { auto [m, h, pr, vm] = tf::isotropic_remeshed(s.polygons(), c, P(), tf::return_index_map);
    REQUIRE(count_true(pr) == 2); require_vmap_survivors(vm, nv, m.points().size()); }
  { auto [m, h, l] = tf::isotropic_remeshed(s.polygons(), c, R()); REQUIRE(l.size() == m.faces().size()); }
  { auto [m, h, l, vm] = tf::isotropic_remeshed(s.polygons(), c, R(), tf::return_index_map);
    REQUIRE(l.size() == m.faces().size()); require_vmap_survivors(vm, nv, m.points().size()); }
  { auto [m, h, l, pr] = tf::isotropic_remeshed(s.polygons(), c, R(), P());
    REQUIRE(l.size() == m.faces().size()); REQUIRE(pr.size() == m.points().size()); }
  { auto [m, h, l, pr, vm] = tf::isotropic_remeshed(s.polygons(), c, R(), P(), tf::return_index_map);
    REQUIRE(l.size() == m.faces().size()); REQUIRE(count_true(pr) == 2);
    require_vmap_survivors(vm, nv, m.points().size()); }
  // target-length variants
  { auto [m, h] = tf::isotropic_remeshed(s.polygons(), mel); REQUIRE(m.faces().size() > 0); }
  { auto [m, h, vm] = tf::isotropic_remeshed(s.polygons(), mel, tf::return_index_map); (void)vm; }
  { auto [m, h, pr] = tf::isotropic_remeshed(s.polygons(), mel, P()); (void)pr; }
  { auto [m, h, l, pr, vm] = tf::isotropic_remeshed(s.polygons(), mel, R(), P(), tf::return_index_map);
    (void)l; (void)pr; (void)vm; }
}
