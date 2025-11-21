/**
 * Benchmark: Isocontours with libigl
 *
 * Measures time to extract isocontours from a scalar field on a mesh
 * using libigl's igl::isolines.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <trueform/trueform.hpp>
#include "test_meshes.hpp"
#include "timing.hpp"
#include "conversions.hpp"
#include <iostream>

#include <igl/isolines.h>

int main() {
    std::cout << "polygons,n_cuts,time_ms\n";

    constexpr int n_samples = 10;
    constexpr int n_cuts_list[] = {1, 2, 4, 8, 16, 32, 64, 128};

    for (const auto& path : benchmark::BENCHMARK_MESHES) {
        auto r_polygons = tf::read_stl<int>(path);

        // Convert to libigl format
        auto V = benchmark::igl::to_igl_vertices(r_polygons.points());
        auto F = benchmark::igl::to_igl_faces(r_polygons.faces());

        // Create scalar field: distance from origin
        Eigen::VectorXd S(V.rows());
        double min_val = std::numeric_limits<double>::max();
        double max_val = std::numeric_limits<double>::lowest();
        for (int i = 0; i < V.rows(); ++i) {
            double val = V.row(i).norm();
            S(i) = val;
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }

        for (int n_cuts : n_cuts_list) {
            // Generate evenly spaced cut values
            Eigen::VectorXd vals(n_cuts);
            for (int i = 0; i < n_cuts; ++i) {
                vals(i) = min_val + (max_val - min_val) * (i + 1) / (n_cuts + 1);
            }

            auto time = benchmark::min_time_of([&]() {
                Eigen::MatrixXd iV;
                Eigen::MatrixXi iE;
                Eigen::VectorXi I;
                ::igl::isolines(V, F, S, vals, iV, iE, I);
                benchmark::do_not_optimize(iV);
                benchmark::do_not_optimize(iE);
            }, n_samples);

            std::cout << r_polygons.faces().size() << "," << n_cuts << "," << time << "\n";
        }
    }

    return 0;
}
