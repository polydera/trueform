/**
 * Decimation quality comparison: TrueForm vs CGAL
 *
 * Decimates the 1M dragon to 50%, computes min angle histogram
 * for both results, and outputs JSON suitable for chart rendering.
 *
 * Usage: ./remesh-quality-decimation
 * Output: JSON array of {angle, tf, cgal} objects (percentages)
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "conversions.hpp"
#include "test_meshes.hpp"
#include <trueform/trueform.hpp>

#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_count_ratio_stop_predicate.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace SMS = CGAL::Surface_mesh_simplification;

static constexpr int N_BINS = 12;
static constexpr float BIN_WIDTH = 5.0f; // degrees per bin, 0-60°

template <typename Polygons>
auto compute_min_angle_histogram(const Polygons &polys)
    -> std::array<float, N_BINS> {
  std::array<int, N_BINS> counts{};
  int total = 0;

  for (const auto &tri : polys) {
    auto p0 = tri[0];
    auto p1 = tri[1];
    auto p2 = tri[2];

    // Edge vectors
    float e0x = p1[0] - p0[0], e0y = p1[1] - p0[1], e0z = p1[2] - p0[2];
    float e1x = p2[0] - p0[0], e1y = p2[1] - p0[1], e1z = p2[2] - p0[2];
    float e2x = p2[0] - p1[0], e2y = p2[1] - p1[1], e2z = p2[2] - p1[2];

    float l0 = std::sqrt(e0x * e0x + e0y * e0y + e0z * e0z);
    float l1 = std::sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
    float l2 = std::sqrt(e2x * e2x + e2y * e2y + e2z * e2z);

    if (l0 < 1e-12f || l1 < 1e-12f || l2 < 1e-12f)
      continue;

    // Angles at each vertex via dot product
    float cos_a0 = (e0x * e1x + e0y * e1y + e0z * e1z) / (l0 * l1);
    float cos_a1 = (-e0x * e2x - e0y * e2y - e0z * e2z) / (l0 * l2);
    float cos_a2 = (e1x * (-e2x) + e1y * (-e2y) + e1z * (-e2z)) / (l1 * l2);

    // Clamp for numerical safety
    cos_a0 = std::max(-1.0f, std::min(1.0f, cos_a0));
    cos_a1 = std::max(-1.0f, std::min(1.0f, cos_a1));
    cos_a2 = std::max(-1.0f, std::min(1.0f, cos_a2));

    float a0 = std::acos(cos_a0) * (180.0f / 3.14159265f);
    float a1 = std::acos(cos_a1) * (180.0f / 3.14159265f);
    float a2 = std::acos(cos_a2) * (180.0f / 3.14159265f);

    float min_angle = std::min({a0, a1, a2});
    int bin = std::min(int(min_angle / BIN_WIDTH), N_BINS - 1);
    counts[bin]++;
    total++;
  }

  std::array<float, N_BINS> pct{};
  for (int i = 0; i < N_BINS; ++i)
    pct[i] = 100.0f * counts[i] / total;
  return pct;
}

template <typename SM>
auto compute_min_angle_histogram_cgal(const SM &mesh)
    -> std::array<float, N_BINS> {
  std::array<int, N_BINS> counts{};
  int total = 0;

  for (auto f : mesh.faces()) {
    auto h = mesh.halfedge(f);
    auto p0 = mesh.point(mesh.target(h));
    auto p1 = mesh.point(mesh.target(mesh.next(h)));
    auto p2 = mesh.point(mesh.target(mesh.next(mesh.next(h))));

    float e0x = p1.x() - p0.x(), e0y = p1.y() - p0.y(),
          e0z = p1.z() - p0.z();
    float e1x = p2.x() - p0.x(), e1y = p2.y() - p0.y(),
          e1z = p2.z() - p0.z();
    float e2x = p2.x() - p1.x(), e2y = p2.y() - p1.y(),
          e2z = p2.z() - p1.z();

    float l0 = std::sqrt(e0x * e0x + e0y * e0y + e0z * e0z);
    float l1 = std::sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
    float l2 = std::sqrt(e2x * e2x + e2y * e2y + e2z * e2z);

    if (l0 < 1e-12f || l1 < 1e-12f || l2 < 1e-12f)
      continue;

    float cos_a0 = (e0x * e1x + e0y * e1y + e0z * e1z) / (l0 * l1);
    float cos_a1 = (-e0x * e2x - e0y * e2y - e0z * e2z) / (l0 * l2);
    float cos_a2 = (e1x * (-e2x) + e1y * (-e2y) + e1z * (-e2z)) / (l1 * l2);

    cos_a0 = std::max(-1.0f, std::min(1.0f, cos_a0));
    cos_a1 = std::max(-1.0f, std::min(1.0f, cos_a1));
    cos_a2 = std::max(-1.0f, std::min(1.0f, cos_a2));

    float a0 = std::acos(cos_a0) * (180.0f / 3.14159265f);
    float a1 = std::acos(cos_a1) * (180.0f / 3.14159265f);
    float a2 = std::acos(cos_a2) * (180.0f / 3.14159265f);

    float min_angle = std::min({a0, a1, a2});
    int bin = std::min(int(min_angle / BIN_WIDTH), N_BINS - 1);
    counts[bin]++;
    total++;
  }

  std::array<float, N_BINS> pct{};
  for (int i = 0; i < N_BINS; ++i)
    pct[i] = 100.0f * counts[i] / total;
  return pct;
}

int main() {
  std::string path = benchmark::BENCHMARK_MESHES.back(); // 1M mesh
  auto mesh = tf::read_stl<int>(path);
  auto polys = mesh.polygons();

  std::cerr << "Input: " << polys.size() << " faces\n";

  // TrueForm decimation
  tf::half_edges<int> he(polys);
  auto tagged = polys | tf::tag(he);
  tf::decimate_config<float> config;
  auto [tf_mesh, tf_he] = tf::decimated(tagged, 0.5f, config);
  std::cerr << "TF result: " << tf_mesh.faces_buffer().size() << " faces\n";
  auto tf_hist = compute_min_angle_histogram(tf_mesh.polygons());

  // CGAL decimation
  auto cgal_mesh = benchmark::cgal::to_cgal_mesh(mesh);
  SMS::edge_collapse(cgal_mesh,
      SMS::Edge_count_ratio_stop_predicate<benchmark::cgal::Surface_mesh>(0.5));
  std::cerr << "CGAL result: " << cgal_mesh.number_of_faces() << " faces\n";
  auto cgal_hist = compute_min_angle_histogram_cgal(cgal_mesh);

  // Output JSON
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "[\n";
  for (int i = 0; i < N_BINS; ++i) {
    float angle = i * BIN_WIDTH + BIN_WIDTH / 2;
    std::cout << "  {\"angle\": " << angle
              << ", \"tf\": " << tf_hist[i]
              << ", \"cgal\": " << cgal_hist[i] << "}";
    if (i < N_BINS - 1)
      std::cout << ",";
    std::cout << "\n";
  }
  std::cout << "]\n";

  return 0;
}
