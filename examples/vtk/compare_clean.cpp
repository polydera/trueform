#include "./util/common.hpp"
#include "./util/data_bridge.hpp"
#include "trueform/trueform.hpp"

#include <filesystem>
#include <iostream>
#include <limits>
#include <string_view>

#include "vtkCleanPolyData.h"

int test_clean(vtkPolyData *poly, int n_iters, double abs_tol) {
  int dummy = 0;
  float t_tf = std::numeric_limits<float>::max();
  float t_vtk = std::numeric_limits<float>::max();

  for (int i = 0; i < n_iters; ++i) {
    // --- VTK ---
    tf::tick();
    auto cleaner = vtk_make_unique<vtkCleanPolyData>();
    cleaner->SetInputData(poly);
    cleaner->PointMergingOn();
    cleaner->ConvertLinesToPointsOff();
    cleaner->ConvertPolysToLinesOff();
    cleaner->ConvertStripsToPolysOff();
    cleaner->ToleranceIsAbsoluteOn();
    cleaner->SetAbsoluteTolerance(abs_tol);
    cleaner->Update();
    vtkPolyData *vtk_out = cleaner->GetOutput();
    dummy += static_cast<int>(vtk_out->GetNumberOfPoints());
    dummy += static_cast<int>(vtk_out->GetNumberOfPolys());
    t_vtk = std::min(t_vtk, tf::tock());

    // --- trueform ---
    tf::tick();
    auto triangles = get_triangles(poly); // util/data_bridge.hpp
    if (abs_tol > 0.0) {
      auto tf_out = tf::cleaned<int>(triangles, static_cast<float>(abs_tol));
      dummy += static_cast<int>(tf_out.points().size());
      dummy += static_cast<int>(tf_out.faces().size());
    } else {
      auto tf_out = tf::cleaned<int>(triangles);
      dummy += static_cast<int>(tf_out.points().size());
      dummy += static_cast<int>(tf_out.faces().size());
    }
    t_tf = std::min(t_tf, tf::tock());
  }

  std::cout << "--- Clean PolyData ---\n";
  std::cout << "       vtk: " << t_vtk << " ms\n";
  std::cout << "        tf: " << t_tf << " ms\n";
  std::cout << "  speed-up: " << (t_vtk / t_tf) << "\n";
  return dummy;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: program <input.stl|obj> [abs_tolerance]\n";
    return 1;
  }
  if (argc == 2 && (std::string_view(argv[1]) == "-h" ||
                    std::string_view(argv[1]) == "--help")) {
    std::cerr << "Usage: program <input.stl|obj> [abs_tolerance]\n";
    return 1;
  }

  std::filesystem::path path{argv[1]};
  double abs_tol = (argc >= 3) ? std::stod(argv[2]) : 1.e-6;

  auto poly = read_mesh(argv[1]); // util/common.hpp / data_bridge.hpp

  std::cout << "number of polygons: " << poly->GetNumberOfPolys() << "\n";
  std::cout << "number of points:   " << poly->GetNumberOfPoints() << "\n";

  int dummy = 0;
  dummy += test_clean(poly.get(), /*n_iters=*/10, abs_tol);
  return dummy;
}
