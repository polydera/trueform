#include "./util/common.hpp"
#include "./util/data_bridge.hpp"
#include "trueform/trueform.hpp"
#include "vtkIntersectionPolyDataFilter.h"
#include <filesystem>

auto apply_transform_to_vtk_mesh(
    vtkPolyData *mesh, const tf::transformation<float, 3> &transform) {
  auto out = vtk_make_unique<vtkPolyData>();
  out->DeepCopy(mesh);
  tf::parallel_apply(get_points(out.get()),
                     [&](auto &&pt) { pt = transform.transform_point(pt); });
  return out;
}

int run_intersections(const std::string &file_name, int n_iters) {
  std::cout << "\n--- Running Intersection Curve Benchmark (" << n_iters
            << " iterations) ---\n";
  // Load CGAL mesh
  auto poly0 = read_mesh(file_name);
  std::cout << "Number of Polygons: " << poly0->GetNumberOfPolys() << std::endl;

  auto [raw_points, raw_faces] = tf::examples::read_mesh(file_name.c_str());
  auto polygons =
      tf::make_polygons(tf::make_blocked_range<3>(raw_faces),
                        tf::make_points<3>(raw_points).as<double>());
  // Build trueform tree once
  tf::tree<int, float, 3> tree_tf;
  tree_tf.build(polygons, tf::config_tree(4, 4));
  tf::face_membership<int> fe;
  fe.build(polygons);
  tf::manifold_edge_link<int, 3> mel;
  mel.build(polygons.faces(), fe);

  float time_vtk = 0.0f;
  float time_tf = 0.0f;
  int dummy_out = 0; // To prevent compiler from optimizing away results

  for (int i = 0; i < n_iters; ++i) {
    // Print progress indicator that updates on the same line
    int progress = static_cast<int>(((i + 1.0) / n_iters) * 100.0);
    std::cout << "\rProgress: " << progress << "%" << std::flush;

    // Generate a new random transformation for each iteration
    // guaranteeing intersections
    auto id0 = tf::random<int>(0, polygons.points().size() - 1);
    auto pt0 = polygons.points()[id0];
    auto id1 = tf::random<int>(0, polygons.points().size() - 1);
    auto pt1 = polygons.points()[id1];
    auto transform = tf::transformed(
        tf::make_transformation_from_translation(-pt1.as_vector_view()),
        tf::random_transformation(pt0.as_vector_view()));
    auto poly1 = apply_transform_to_vtk_mesh(poly0.get(), transform);

    vtkNew<vtkIntersectionPolyDataFilter> intersection_filter;
    intersection_filter->SetInputData(0, poly0.get());
    intersection_filter->SetInputData(1, poly1.get());
    intersection_filter->SetCheckInput(false);
    intersection_filter->SetCheckMesh(false);
    intersection_filter->SetSplitFirstOutput(false);
    intersection_filter->SetSplitSecondOutput(false);

    tf::tick();
    intersection_filter->Update(); // This is the main computation
    time_vtk += tf::tock();

    auto form0 = tf::make_form(tf::make_frame(transform), tree_tf,
                               polygons | tf::tag(fe) | tf::tag(mel));
    auto form1 = tf::make_form(tree_tf, polygons | tf::tag(fe) | tf::tag(mel));
    tf::tick();
    tf::polygons_intersections<int, double, 3> fi;
    fi.build(form0, form1);
    auto edges = tf::make_intersection_edges(fi);
    time_tf += tf::tock();
    dummy_out += edges.size();
    dummy_out += intersection_filter->GetOutput()->GetNumberOfLines();
  }

  std::cout << "\n\n--- Intersection Curve Results ---\n";
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Average trueform time: " << (time_tf / n_iters) << " ms\n";
  std::cout << "Average VTK time:     " << (time_vtk / n_iters) << " ms\n";
  std::cout << "Speed-up:              " << (time_vtk / time_tf) << "x\n";

  return dummy_out;
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

  int dummy = 0;
  dummy += run_intersections(argv[1], 10);
  return dummy;
}
