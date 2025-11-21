/**
 * Benchmark: Boundary paths with VTK
 *
 * Measures time to extract boundary edges for triangle meshes
 * using VTK's vtkFeatureEdges.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <trueform/trueform.hpp>
#include "test_meshes.hpp"
#include "timing.hpp"
#include "conversions.hpp"
#include <iostream>

#include <vtkFeatureEdges.h>
#include <vtkSmartPointer.h>

int main() {
    std::cout << "polygons,time_ms\n";

    for (const auto& path : benchmark::BENCHMARK_MESHES) {
        auto r_polygons = tf::read_stl<int>(path);

        // Convert to VTK polydata
        auto polydata = benchmark::vtk::to_vtk_polydata(r_polygons);

        auto time = benchmark::min_time_of([&]() {
            auto featureEdges = vtkSmartPointer<vtkFeatureEdges>::New();
            featureEdges->SetInputData(polydata);
            featureEdges->BoundaryEdgesOn();
            featureEdges->FeatureEdgesOff();
            featureEdges->ManifoldEdgesOff();
            featureEdges->NonManifoldEdgesOff();
            featureEdges->Update();
            benchmark::do_not_optimize(featureEdges->GetOutput());
        });

        std::cout << r_polygons.faces().size() << "," << time << "\n";
    }

    return 0;
}
