/**
 * Benchmark: Mesh-mesh intersection curves with VTK
 *
 * Measures time to compute intersection curves between two meshes
 * using VTK's vtkIntersectionPolyDataFilter.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <trueform/trueform.hpp>
#include "test_meshes.hpp"
#include "timing.hpp"
#include "conversions.hpp"
#include <iostream>

#include <vtkIntersectionPolyDataFilter.h>
#include <vtkSmartPointer.h>

int main() {
    std::cout << "polygons0,polygons1,time_ms\n";

    constexpr int n_samples = 100;

    for (const auto& path : benchmark::BENCHMARK_MESHES) {
        auto r_polygons = tf::read_stl<int>(path);
        auto points = r_polygons.points();

        // Create mesh1
        auto mesh1 = benchmark::vtk::to_vtk_polydata(r_polygons);

        // Calculate diagonal length for transform generation
        auto l = tf::aabb_from(r_polygons.polygons()).diagonal().length();

        vtkSmartPointer<vtkPolyData> mesh2;
        auto time_ms = benchmark::mean_time_of(
            [&]() {
                // Pick a random "pivot" point for the random transformation
                auto pivot_idx = tf::random<int>(0, static_cast<int>(points.size()));
                auto pivot = points[pivot_idx];
                auto new_origin_id = tf::random<int>(0, static_cast<int>(points.size()));
                auto new_origin = points[new_origin_id];
                auto transform = tf::random_transformation_at(pivot, new_origin);

                // Transform TrueForm mesh, then convert to VTK
                tf::polygons_buffer<int, float, 3, 3> transformed;
                transformed.faces_buffer() = r_polygons.faces_buffer();
                transformed.points_buffer().allocate(points.size());
                tf::parallel_transform(points, transformed.points(), [&](auto pt) {
                    return tf::transformed(pt, transform);
                });

                mesh2 = benchmark::vtk::to_vtk_polydata(transformed);
            },
            [&]() {
                auto intersectionFilter = vtkSmartPointer<vtkIntersectionPolyDataFilter>::New();
                intersectionFilter->SetInputData(0, mesh1);
                intersectionFilter->SetInputData(1, mesh2);
                intersectionFilter->Update();
                benchmark::do_not_optimize(intersectionFilter->GetOutput());
            },
            n_samples);

        std::cout << r_polygons.faces().size() << "," << r_polygons.faces().size() << "," << time_ms << "\n";
    }

    return 0;
}
