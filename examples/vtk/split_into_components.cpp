#include "./util/common.hpp"
#include "./util/data_bridge.hpp"
#include "trueform/trueform.hpp"
#include <vtkMatrix4x4.h>
#include <vtkNew.h>
#include <vtkOpenGLActor.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkPolyData.h>
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
  auto polys = get_triangles(poly.get());

  auto polygons = tf::concatenated(polys, polys, polys);

  tf::face_membership<int> fm;
  fm.build(polygons.polygons());
  tf::face_link<int> fl;
  fl.build(polygons.faces(), fm);

  tf::buffer<int> labels;
  labels.allocate(polygons.size());
  int n_components =
      tf::label_connected_components<int>(labels, tf::make_applier(fl));
  std::cout << "n_components: " << n_components << std::endl;

  auto comps = tf::split_into_components(polygons.polygons(), labels);
  std::cout << "v size: " << comps.size() << std::endl;

  // --- Create render pipeline once (so we can Start() after the loop) ---
  vtkNew<vtkRenderer> renderer;
  vtkNew<vtkRenderWindow> renderWindow;
  vtkNew<vtkRenderWindowInteractor> interactor;
  renderWindow->AddRenderer(renderer.GetPointer());
  interactor->SetRenderWindow(renderWindow.GetPointer());
  renderer->SetBackground(27.0 / 255.0, 43.0 / 255.0, 52.0 / 255.0);

  // --- Use the mesh scale to space components nicely ---
  const float spacing =
      static_cast<float>(tf::aabb_from(polys).diagonal().length()) * 1.25f;

  // Simple grid layout
  const int cols = 5; // tweak if you like; spacing handles scale
  int idx = 0;

  for (const auto &mesh : comps) {
    // Convert to vtkPolyData
    auto p = to_polydata(mesh);

    // Mapper + actor
    vtkNew<vtkOpenGLPolyDataMapper> mapper;
    mapper->SetInputData(p.get());

    vtkNew<vtkOpenGLActor> actor;
    actor->SetMapper(mapper.GetPointer());

    // Place actors on a grid using 'spacing'
    const int ci = idx % cols;
    const int cj = idx / cols;
    vtkNew<vtkMatrix4x4> T;
    T->Identity();
    set_at(T.GetPointer(), {ci * spacing, cj * spacing, 0.f});
    actor->SetUserMatrix(T.GetPointer());
    ++idx;

    renderer->AddActor(actor.GetPointer());
  }

  renderWindow->Render();
  interactor->Start();
  return 0;
}
