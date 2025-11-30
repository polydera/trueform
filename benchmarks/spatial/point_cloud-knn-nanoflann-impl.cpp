/**
 * Point cloud kNN nanoflann - Implementation
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "point_cloud-knn-nanoflann.hpp"
#include "timing.hpp"
#include <nanoflann.hpp>
#include <trueform/trueform.hpp>
#include <vector>

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

int run_point_cloud_knn_nanoflann_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "bv,points,k,time_ms\n";

  constexpr int max_k = 10;

  for (const auto &path : mesh_paths) {
    auto polygons = tf::read_stl<int>(path);
    auto &points = polygons.points_buffer();

    using Adapter = PointCloudAdapter<float, 3>;
    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, Adapter>, Adapter, 3>;

    Adapter adapter(points.data_buffer().begin(), points.size());
    KDTree tree(3, adapter, {4});

    auto l = tf::aabb_from(polygons.points()).diagonal().length();

    std::array<float, 3> query_point;

    std::vector<size_t> indices(max_k);
    std::vector<float> distances(max_k);

    for (int k = 1; k <= max_k; ++k) {
      auto time = benchmark::mean_time_of(
          [&]() {
            auto idx = tf::random<int>(0, points.size() - 1);
            auto point = polygons.points()[idx];
            auto random_offset = tf::random_vector<float, 3>() * l;
            query_point[0] = point[0] + random_offset[0];
            query_point[1] = point[1] + random_offset[1];
            query_point[2] = point[2] + random_offset[2];
          },
          [&]() {
            nanoflann::KNNResultSet<float> resultSet(k);
            resultSet.init(&indices[0], &distances[0]);
            tree.findNeighbors(resultSet, &query_point[0],
                               nanoflann::SearchParameters());
            benchmark::do_not_optimize(indices);
          },
          n_samples);

      out << "KDTree," << points.size() << "," << k << "," << time << "\n";
    }
  }

  return 0;
}

} // namespace benchmark
