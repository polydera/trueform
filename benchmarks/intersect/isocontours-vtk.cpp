/**
 * Benchmark: Isocontours with VTK
 *
 * Measures time to extract isocontours from a scalar field on a mesh
 * using VTK's vtkContourFilter.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <trueform/trueform.hpp>
#include "test_meshes.hpp"
#include "timing.hpp"
#include "conversions.hpp"
#include <iostream>

#include <vtkContourFilter.h>
#include <vtkSmartPointer.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>

int main() {
    std::cout << "polygons,n_cuts,time_ms\n";

    constexpr int n_samples = 10;
    constexpr int n_cuts_list[] = {1, 2, 4, 8, 16, 32, 64, 128};

    for (const auto& path : benchmark::BENCHMARK_MESHES) {
        auto r_polygons = tf::read_stl<int>(path);
        auto polydata = benchmark::vtk::to_vtk_polydata(r_polygons);

        // Create scalar field: distance from origin
        auto scalars = vtkSmartPointer<vtkFloatArray>::New();
        scalars->SetName("Distance");
        scalars->SetNumberOfTuples(polydata->GetNumberOfPoints());

        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        for (vtkIdType i = 0; i < polydata->GetNumberOfPoints(); ++i) {
            double pt[3];
            polydata->GetPoint(i, pt);
            float dist = std::sqrt(pt[0]*pt[0] + pt[1]*pt[1] + pt[2]*pt[2]);
            scalars->SetValue(i, dist);
            min_val = std::min(min_val, dist);
            max_val = std::max(max_val, dist);
        }
        polydata->GetPointData()->SetScalars(scalars);

        for (int n_cuts : n_cuts_list) {
            // Generate evenly spaced cut values
            std::vector<float> cut_values(n_cuts);
            for (int i = 0; i < n_cuts; ++i) {
                cut_values[i] = min_val + (max_val - min_val) * (i + 1) / (n_cuts + 1);
            }

            auto time = benchmark::min_time_of([&]() {
                auto contourFilter = vtkSmartPointer<vtkContourFilter>::New();
                contourFilter->SetInputData(polydata);
                contourFilter->SetNumberOfContours(n_cuts);
                for (int i = 0; i < n_cuts; ++i) {
                    contourFilter->SetValue(i, cut_values[i]);
                }
                contourFilter->Update();
                benchmark::do_not_optimize(contourFilter->GetOutput());
            }, n_samples);

            std::cout << r_polygons.faces().size() << "," << n_cuts << "," << time << "\n";
        }
    }

    return 0;
}
