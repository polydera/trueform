#include "./util/common.hpp"
#include "./util/data_bridge.hpp"
#include "trueform/trueform.hpp"
#include "vtkFeatureEdges.h"
#include <filesystem>

int test_boundary(vtkPolyData *poly, int n_iters) {
  int dummy = 0;
  float t_tf = std::numeric_limits<float>::max();
  float t_vtk = std::numeric_limits<float>::max();
  for (int i = 0; i < n_iters; ++i) {
    tf::tick();
    auto featureEdges = vtk_make_unique<vtkFeatureEdges>();
    featureEdges->SetInputData(poly);
    featureEdges->BoundaryEdgesOn();
    featureEdges->FeatureEdgesOff();
    featureEdges->ManifoldEdgesOff();
    featureEdges->NonManifoldEdgesOff();
    featureEdges->Update();
    vtkPolyData *boundaryEdges = featureEdges->GetOutput();
    dummy += boundaryEdges->GetNumberOfLines();
    t_vtk = std::min(t_vtk, tf::tock());
    tf::tick();
    auto boundary_edges = tf::make_boundary_edges(get_triangles(poly));
    t_tf = std::min(t_tf, tf::tock());
    dummy += boundary_edges.size();
  }
  std::cout << "--- Boundary Edges ---" << std::endl;
  std::cout << "       vtk: " << t_vtk << " ms" << std::endl;
  std::cout << "        tf: " << t_tf << " ms" << std::endl;
  std::cout << "  speed-up: " << (t_vtk / t_tf) << std::endl;
  return dummy;
}

int test_non_manifold(vtkPolyData *poly, int n_iters) {
  int dummy = 0;
  float t_tf = std::numeric_limits<float>::max();
  float t_vtk = std::numeric_limits<float>::max();
  for (int i = 0; i < n_iters; ++i) {
    tf::tick();
    auto featureEdges = vtk_make_unique<vtkFeatureEdges>();
    featureEdges->SetInputData(poly);
    featureEdges->BoundaryEdgesOff();
    featureEdges->FeatureEdgesOff();
    featureEdges->ManifoldEdgesOn();
    featureEdges->Update();
    vtkPolyData *nonManifoldEdges = featureEdges->GetOutput();
    dummy += nonManifoldEdges->GetNumberOfLines();
    t_vtk = std::min(t_vtk, tf::tock());
    tf::tick();
    auto non_manifold_edges = tf::make_non_manifold_edges(get_triangles(poly));
    t_tf = std::min(t_tf, tf::tock());
    dummy += non_manifold_edges.size();
  }
  std::cout << "--- Non-Manifold Edges ---" << std::endl;
  std::cout << "       vtk: " << t_vtk << " ms" << std::endl;
  std::cout << "        tf: " << t_tf << " ms" << std::endl;
  std::cout << "  speed-up: " << (t_vtk / t_tf) << std::endl;
  return dummy;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: program <input1.stl>\n";
    return 1;
  } else if (argc == 2 && (std::string_view(argv[1]) == "-h" ||
                           std::string_view(argv[1]) == "--help")) {
    std::cerr << "Usage: program <input1.stl>\n";
    return 1;
  }
  std::filesystem::path path{argv[1]};

  if (!path.has_extension() || path.extension() != ".stl") {
    std::cerr << "Skipping file " << path.filename() << ": not an .stl file\n";
    return 1;
  }

  auto poly = readSTL(argv[1]);
  int dummy = 0;
  std::cout << "number of polygons: " << poly->GetNumberOfPolys() << std::endl;
  dummy += test_boundary(poly.get(), 10);
  dummy += test_non_manifold(poly.get(), 10);
  return dummy;
}
