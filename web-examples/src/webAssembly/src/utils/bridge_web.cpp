#include "trueform/random.hpp"
#include "trueform/trueform.hpp"

#include <emscripten/bind.h>
#include <emscripten/val.h>

class MeshObject {
public:
    MeshObject() {
        for (int i = 0; i < 16; ++i) {
            if (i % 5 == 0)
                matrix[i] = 1;
            else
                matrix[i] = 0;
        }
    };
    ~MeshObject() = default;

    tf::polygons_buffer<int, float, 3, 3> polyObject;
    tf::curves_buffer<int, float, 3> curvesObject;
    std::array<double, 16> matrix;


    emscripten::val GetPoints() {
        return emscripten::val(emscripten::typed_memory_view(polyObject.points_buffer().data_buffer().size(), polyObject.points_buffer().data_buffer().begin()));
    }
    emscripten::val GetPolys() {
        return emscripten::val(emscripten::typed_memory_view(polyObject.faces_buffer().data_buffer().size(), polyObject.faces_buffer().data_buffer().begin()));
    }

    emscripten::val GetMatrix() {
        return emscripten::val(emscripten::typed_memory_view(matrix.size(), matrix.data()));

    }
    // CURVE LINES
    // tf::curves_buffer<int, float, 3> cb;
    // auto &ids = cb.paths_buffer().data_buffer(); // ids
    // auto &offsets = cb.paths_buffer().offsets_buffer();// [0, a, b, c, ..., ids.size()]
    // ids for curve[0] --- ids[offsets[0]]... ids[offsets[1]-1]
    // for (auto line_ids: cb.paths_buffer()) {
    // auto points_for_line = tf::make_indirect_range(line_ids, cb.points());
    // }
};


