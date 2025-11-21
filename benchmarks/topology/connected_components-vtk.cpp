/**
 * Benchmark: Connected components with VTK
 *
 * Measures time to compute connected component labels for triangle meshes
 * using VTK's vtkPolyDataConnectivityFilter.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <trueform/trueform.hpp>
#include "test_meshes.hpp"
#include "timing.hpp"
#include "conversions.hpp"
#include <iostream>

#include <vtkPolyDataConnectivityFilter.h>
#include <vtkSmartPointer.h>

int main() {
    std::cout << "polygons,time_ms\n";

    for (const auto& path : benchmark::BENCHMARK_MESHES) {
        auto r_polygons = tf::read_stl<int>(path);

        // Convert to VTK polydata
        auto polydata = benchmark::vtk::to_vtk_polydata(r_polygons);

        auto time = benchmark::min_time_of([&]() {
            auto connectivity = vtkSmartPointer<vtkPolyDataConnectivityFilter>::New();
            connectivity->SetInputData(polydata);
            connectivity->SetExtractionModeToAllRegions();
            connectivity->ColorRegionsOn();
            connectivity->Update();
            benchmark::do_not_optimize(connectivity->GetOutput());
        });

        std::cout << r_polygons.faces().size() << "," << time << "\n";
    }

    return 0;
}
