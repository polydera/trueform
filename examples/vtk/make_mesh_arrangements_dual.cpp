#include "./util/common.hpp"
#include "./util/data_bridge.hpp"
#include "trueform/trueform.hpp"
#include <vtkMatrix4x4.h>
#include <vtkNew.h>
#include <vtkOpenGLActor.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>

auto set_at(vtkMatrix4x4 *mat, tf::vector<float, 3> at) -> void {
  auto tr = tf::random_transformation(at);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      mat->Element[i][j] = tr(i, j);
  mat->Modified();
}

int main(int argc, char *argv[]) {
  if (argc < 2 || (argc == 2 && (std::string_view(argv[1]) == "-h" ||
                                 std::string_view(argv[1]) == "--help"))) {
    std::cerr << "Usage: program <input.stl|obj>\n";
    return 1;
  }

  // Load + normalize input
  auto poly = read_mesh(argv[1]);
  auto polygons = get_triangles(poly.get());

  auto id0 = tf::random<int>(0, polygons.points().size() - 1);
  auto pt0 = polygons.points()[id0];
  auto id1 = tf::random<int>(0, polygons.points().size() - 1);
  auto pt1 = polygons.points()[id1];

  auto transform = tf::transformed(
      tf::make_transformation_from_translation(-pt1.as_vector_view()),
      tf::random_transformation(pt0.as_vector_view()));

  tf::tree<int, float, 3> tree_tf;
  tree_tf.build(polygons, tf::config_tree(4, 4));

  tf::face_membership<int> fm;
  fm.build(polygons);

  tf::manifold_edge_link<int, 3> mel;
  mel.build(polygons.faces(), fm);

  auto form0 = tf::make_form(tree_tf, polygons | tf::tag(mel) | tf::tag(fm));
  auto form1 = tf::make_form(tf::make_frame(transform), tree_tf,
                             polygons | tf::tag(mel) | tf::tag(fm));

  auto [meshes, labels, containments] =
      tf::make_mesh_arrangements(form0, form1);

  // --- VTK setup ---
  vtkNew<vtkRenderer> renderer;
  vtkNew<vtkRenderWindow> renderWindow;
  vtkNew<vtkRenderWindowInteractor> interactor;
  renderWindow->AddRenderer(renderer.GetPointer());
  interactor->SetRenderWindow(renderWindow.GetPointer());
  renderer->SetBackground(27.0 / 255.0, 43.0 / 255.0, 52.0 / 255.0);

  // --- dynamic columns based on sqrt(#meshes) ---
  const int N = static_cast<int>(meshes.size());
  const int cols = std::max(
      1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(N)))));

  // --- per-row cursoring ---
  float x_cursor = 0.0f;
  float y_cursor = 0.0f;
  float row_max_h = 0.0f;
  int col = 0;

  // --- color helpers ---
  auto shade = [](const std::array<double, 3> &c, double s) {
    return std::array<double, 3>{std::clamp(c[0] * s, 0.0, 1.0),
                                 std::clamp(c[1] * s, 0.0, 1.0),
                                 std::clamp(c[2] * s, 0.0, 1.0)};
  };

  // Base colors per came_from: 0 → blue-ish, 1 → orange-ish
  const std::array<double, 3> base0{0.18, 0.55, 0.95}; // came_from == 0
  const std::array<double, 3> base1{0.98, 0.56, 0.22}; // came_from == 1
  // Shades: outside = darker, inside = brighter
  constexpr double kOutsideMul = 0.60;
  constexpr double kInsideMul = 1.20;


  for (const auto &[mesh, came_from, containment] :
       tf::zip(meshes, labels, containments)) {
    // AABB-driven spacing for THIS piece
    const auto aabb = tf::aabb_from(mesh.polygons());
    const auto d = aabb.diagonal();                       // vec3
    const float stepX = static_cast<float>(d[0]) * 1.20f; // width + 20% pad
    const float stepY = static_cast<float>(d[1]) * 1.20f; // height + 20% pad

    // Convert to vtkPolyData
    auto p = to_polydata(mesh);

    // Mapper + actor
    vtkNew<vtkOpenGLPolyDataMapper> mapper;
    mapper->SetInputData(p.get());
    vtkNew<vtkOpenGLActor> actor;
    actor->SetMapper(mapper.GetPointer());

    // Choose color from (came_from, containment)
    const bool from1 = (came_from == 1);
    const bool is_inside = (containment == tf::strict_containment::inside);

    const auto &base = from1 ? base1 : base0;
    const auto rgb =
        is_inside ? shade(base, kInsideMul) : shade(base, kOutsideMul);
    actor->GetProperty()->SetColor(rgb[0], rgb[1], rgb[2]);

    // Place at current cursor
    vtkNew<vtkMatrix4x4> T;
    T->Identity();
    set_at(T.GetPointer(), {x_cursor, y_cursor, 0.f});
    actor->SetUserMatrix(T.GetPointer());
    renderer->AddActor(actor.GetPointer());

    // Advance within row
    x_cursor += stepX;
    row_max_h = std::max(row_max_h, stepY);
    ++col;

    // Wrap to next row
    if (col >= cols) {
      col = 0;
      x_cursor = 0.0f;
      y_cursor += row_max_h;
      row_max_h = 0.0f;
    }
  }

  renderWindow->Render();
  interactor->Start();
  return 0;
}
