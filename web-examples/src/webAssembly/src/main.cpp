#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "main.h"
#include "boolean_web.h"
#include "utils/bridge_web.h"
#include "utils/cursor_interactor_interface.h"


auto OnLeftButtonUp() {
    return interactor->OnLeftButtonUp();
}

auto OnLeftButtonDown() {
    return interactor->OnLeftButtonDown();
}

auto OnMouseMove(std::array<float, 3> origin, std::array<float, 3> direction, std::array<float, 3> cameraPosition, std::array<float, 3> cameraFocalPoint) {
    return interactor->OnMouseMove(origin, direction, cameraPosition, cameraFocalPoint);
}

auto OnKeyPress(std::string key) {
    return interactor->OnKeyPress(key);
}

auto GetMeshOnIdx(int i) -> mesh_object * {
    return interactor->get_actors()[i].get();
}

auto GetResultMesh() -> mesh_object * {
    if(auto pI = dynamic_cast<cursor_interactor*>(interactor.get())) {
      return pI->result_mesh.get();
    }
    return nullptr;
}

auto GetCurveMesh() -> mesh_object * {
    if(auto pI = dynamic_cast<cursor_interactor*>(interactor.get())) {
        return pI->curve_mesh.get();
    }
    return nullptr;
}

auto GetAverageTime() {
    return interactor->mTime;
}


EMSCRIPTEN_BINDINGS(boolean) {
  emscripten::function("run_main", &run_main);
  emscripten::function("GetAverageTime", &GetAverageTime);
  emscripten::function("OnLeftButtonUp", &OnLeftButtonUp);
  emscripten::function("OnLeftButtonDown", &OnLeftButtonDown);
  emscripten::function("OnMouseMove", &OnMouseMove);
  emscripten::function("OnKeyPress", &OnKeyPress);
  emscripten::function("GetMeshOnIdx", &GetMeshOnIdx, emscripten::allow_raw_pointers());
  emscripten::function("GetResultMesh", &GetResultMesh, emscripten::allow_raw_pointers());
  emscripten::function("GetCurveMesh", &GetCurveMesh, emscripten::allow_raw_pointers());
}

EMSCRIPTEN_BINDINGS(ArrayFloat3) {
    emscripten::value_array<std::array<float, 3>>("ArrayFloat3")
        .element(emscripten::index<0>())
        .element(emscripten::index<1>())
        .element(emscripten::index<2>());
}
EMSCRIPTEN_BINDINGS(ArrayDouble16) {
    emscripten::value_array<std::array<double, 16>>("ArrayDouble16")
        .element(emscripten::index<0>())
        .element(emscripten::index<1>())
        .element(emscripten::index<2>())
        .element(emscripten::index<3>())
        .element(emscripten::index<4>())
        .element(emscripten::index<5>())
        .element(emscripten::index<6>())
        .element(emscripten::index<7>())
        .element(emscripten::index<8>())
        .element(emscripten::index<9>())
        .element(emscripten::index<10>())
        .element(emscripten::index<11>())
        .element(emscripten::index<12>())
        .element(emscripten::index<13>())
        .element(emscripten::index<14>())
        .element(emscripten::index<15>());
}


EMSCRIPTEN_BINDINGS(mesh_object) {
    emscripten::class_<mesh_object>("mesh_object")
        .smart_ptr<std::shared_ptr<mesh_object>>("mesh_object")
        .function("GetPoints", &mesh_object::GetPoints)
        .function("GetPolys", &mesh_object::GetPolys)
        .function("GetCurvePoints", &mesh_object::GetCurvePoints)
        .function("GetCurveIds", &mesh_object::GetCurveIds)
        .function("GetCurveOffsets", &mesh_object::GetCurveOffsets)
        .property("matrix", &mesh_object::matrix)
        .property("matrix_updated", &mesh_object::matrix_updated)
        .property("polydata_updated", &mesh_object::polydata_updated)
    ;
}