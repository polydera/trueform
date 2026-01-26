/**
 * Point cloud tree building benchmark with nanoflann - Implementation
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "point_cloud-build_tree-nanoflann.hpp"
#include "timing.hpp"
#include <nanoflann.hpp>
#include <trueform/trueform.hpp>

// Adapter for nanoflann to work with TrueForm point data
template <typename Real, int Dims> struct PointCloudAdapter {
  const Real *data;
  size_t n_points;

  PointCloudAdapter(const Real *data_, size_t n) : data(data_), n_points(n) {}

  inline size_t kdtree_get_point_count() const { return n_points; }

  inline Real kdtree_get_pt(const size_t idx, const size_t dim) const {
    return data[idx * Dims + dim];
  }

  template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
};

namespace benchmark {

int run_point_cloud_build_tree_nanoflann_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "bv,points,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto polygons = tf::read_stl<int>(path);
    auto &points = polygons.points_buffer();

    using Adapter = PointCloudAdapter<float, 3>;
    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, Adapter>, Adapter, 3>;

    auto time = benchmark::min_time_of(
        [&]() {
          Adapter adapter(points.data_buffer().begin(), points.size());
          KDTree tree(3, adapter, {4});
          benchmark::do_not_optimize(tree);
        },
        n_samples);

    out << "KDTree," << points.size() << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
