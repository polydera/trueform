/**
 * Benchmark: Point cloud tree building with nanoflann
 *
 * Measures time to build spatial acceleration structure (KD-tree)
 * on point clouds of varying sizes using nanoflann library.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include <trueform/trueform.hpp>
#include "test_meshes.hpp"
#include "timing.hpp"
#include <nanoflann.hpp>
#include <iostream>

// Adapter for nanoflann to work with TrueForm point data
template <typename Real, int Dims>
struct PointCloudAdapter {
    const Real* data;
    size_t n_points;

    PointCloudAdapter(const Real* data_, size_t n) : data(data_), n_points(n) {}

    inline size_t kdtree_get_point_count() const { return n_points; }

    inline Real kdtree_get_pt(const size_t idx, const size_t dim) const {
        return data[idx * Dims + dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

int main() {
    std::cout << "points,time_ms\n";

    for (const auto& path : benchmark::BENCHMARK_MESHES) {
        auto polygons = tf::read_stl<int>(path);
        auto &points = polygons.points_buffer();

        using Adapter = PointCloudAdapter<float, 3>;
        using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<float, Adapter>,
            Adapter,
            3>;

        auto time = benchmark::min_time_of([&]() {
            Adapter adapter(points.data_buffer().begin(), points.size());
            KDTree tree(3, adapter, {4});  // max leaf size = 10
            benchmark::do_not_optimize(tree);
        });

        std::cout << points.size() << "," << time << "\n";
    }

    return 0;
}
