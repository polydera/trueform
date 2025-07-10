#include "./util/common.hpp"
#include "./util/data_bridge.hpp"
#include "trueform/trueform.hpp"
#include "vtkContourFilter.h"
#include "vtkFloatArray.h"
#include "vtkPointData.h"
#include <filesystem>

int dummy = 0;

auto run_test(vtkPolyData *poly, int n_contours, tf::plane<float, 3> plane) {

  auto array = vtk_make_unique<vtkFloatArray>();
  array->SetNumberOfComponents(1);
  array->SetNumberOfTuples(poly->GetNumberOfPoints());
  auto scalars =
      tf::make_range(array->GetPointer(0), array->GetNumberOfTuples());
  auto polygons = get_triangles(poly);
  tf::parallel_transform(polygons.points(), scalars, tf::distance_f(plane));
  auto min = *std::min_element(scalars.begin(), scalars.end());
  auto max = *std::max_element(scalars.begin(), scalars.end());
  std::vector<float> cut_values;
  float step = (max - min) / (n_contours + 1);
  for (int i = 0; i <= n_contours; ++i) {
    cut_values.push_back(step * i);
  }

  poly->GetPointData()->SetScalars(array.get());

  tf::tick();
  vtkNew<vtkContourFilter> contour;
  contour->SetInputData(poly);
  contour->SetValue(0, 0);
  for (auto [i, e] : tf::enumerate(cut_values))
    contour->SetValue(i, e);
  contour->Update();
  auto stripper = vtk_make_unique<vtkStripper>();
  stripper->SetInputData(contour->GetOutput());
  stripper->Update();
  float t_vtk = tf::tock();
  dummy += stripper->GetOutput()->GetNumberOfLines();
  //
  tf::tick();
  tf::scalar_field_intersections<int, float, 3> sfi;
  sfi.build_many(polygons, scalars, cut_values);
  auto paths = tf::make_intersection_paths(sfi, polygons, scalars, tf::make_range(cut_values));
  auto curves = tf::make_curves(paths, sfi.intersection_points());
  float t_tf = tf::tock();
  dummy += curves.paths().size();
  //
  return std::make_pair(t_vtk, t_tf);
}

auto run_test(vtkPolyData *poly, int n_contours, int n_iters) {
  float t_vtk = 0;
  float t_tf = 0;
  auto polygons = get_triangles(poly);
  for (int i = 0; i < n_iters; ++i) {
    auto plane = tf::make_plane(
        tf::normalized(tf::random_vector<float, 3>()),
        polygons.points()[tf::random<int>(0, polygons.points().size() - 1)]);
    std::cout << "\rRunning on sample " << (i + 1) << " out of " << n_iters
              << "...      " << std::flush;
    auto [t_vtk_, t_tf_] = run_test(poly, n_contours, plane);
    t_vtk += t_vtk_;
    t_tf += t_tf_;
  }
  t_vtk /= n_iters;
  t_tf /= n_iters;

  std::cout << "\r--- Compute " << n_contours << " Contours ---       " << std::endl;
  std::cout << "       vtk: " << t_vtk << " ms" << std::endl;
  std::cout << "        tf: " << t_tf << " ms" << std::endl;
  std::cout << "  speed-up: " << (t_vtk / t_tf) << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: program <input1.stl|obj>\n";
    return 1;
  } else if (argc == 2 && (std::string_view(argv[1]) == "-h" ||
                           std::string_view(argv[1]) == "--help")) {
    std::cerr << "Usage: program <input1.stl|obj>\n";
    return 1;
  }
  std::filesystem::path path{argv[1]};

  auto poly = read_mesh(argv[1]);
  std::cout << "Number of polygons: " << poly->GetNumberOfPolys() << std::endl;
  int n_iters = 100;
  std::vector<int> n_countours{{1, 10, 50, 100, 500, 1000}};
  std::cout << "--- Scalar Field Intersection Curves ---" << std::endl;
  for (auto n : n_countours)
    run_test(poly.get(), n, n_iters);

  return dummy;
}
