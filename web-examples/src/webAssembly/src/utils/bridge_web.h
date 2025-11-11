#pragma once
#include "trueform/random.hpp"
#include "trueform/trueform.hpp"

#include <emscripten/bind.h>
#include <emscripten/val.h>

class mesh_object {
public:
    mesh_object() {
        for (int i = 0; i < 16; ++i) {
            if (i % 5 == 0)
                matrix[i] = 1;
            else
                matrix[i] = 0;
        }
    };

    bool matrix_updated = true;
    bool polydata_updated = true;

    tf::polygons_buffer<int, float, 3, 3> poly_object;
    tf::curves_buffer<int, float, 3> curves_object;
    std::array<double, 16> matrix;

    auto set_polydata(tf::polygons_buffer<int, float, 3, 3> polydata) -> void {
        poly_object = std::move(polydata);
        polydata_updated = true;
    }
    auto set_curves_object(tf::curves_buffer<int, float, 3> polydata) -> void {
        curves_object = std::move(polydata);
        polydata_updated = true;
    }

    auto GetPoints() -> emscripten::val {
        polydata_updated = false;
        return emscripten::val(emscripten::typed_memory_view(poly_object.points_buffer().data_buffer().size(), poly_object.points_buffer().data_buffer().begin()));
    }
    auto GetPolys() -> emscripten::val {
        return emscripten::val(emscripten::typed_memory_view(poly_object.faces_buffer().data_buffer().size(), poly_object.faces_buffer().data_buffer().begin()));
    }

    auto GetCurvePoints() -> emscripten::val {
        polydata_updated = false;
        return emscripten::val(emscripten::typed_memory_view(curves_object.points_buffer().data_buffer().size(), curves_object.points_buffer().data_buffer().begin()));
    }
    auto GetCurveIds() -> emscripten::val {
        polydata_updated = false;
        return emscripten::val(emscripten::typed_memory_view(curves_object.paths_buffer().data_buffer().size(), curves_object.paths_buffer().data_buffer().begin()));
    }
    auto GetCurveOffsets() -> emscripten::val {
        polydata_updated = false;
        return emscripten::val(emscripten::typed_memory_view(curves_object.paths_buffer().offsets_buffer().size(), curves_object.paths_buffer().offsets_buffer().begin()));
    }

    auto GetMatrix() -> emscripten::val {
        matrix_updated = false;
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