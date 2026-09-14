// The layer's front doors. A caller's first line is one of these, and nothing
// else in the repository compiles them: the syntax sweep walks translation
// units, so a header no unit includes is not in it.
#include <trueform/cpp.hpp>

#include <trueform/cpp/arrangement.hpp>
#include <trueform/cpp/clean.hpp>
#include <trueform/cpp/core.hpp>
#include <trueform/cpp/csg.hpp>
#include <trueform/cpp/geometry.hpp>
#include <trueform/cpp/intersect.hpp>
#include <trueform/cpp/io.hpp>
#include <trueform/cpp/iso.hpp>
#include <trueform/cpp/reindex.hpp>
#include <trueform/cpp/remesh.hpp>
#include <trueform/cpp/spatial.hpp>
#include <trueform/cpp/topology.hpp>

auto main() -> int {
  const tf::polygons_buffer<tf::cpp::default_index_t, float, 3, 3> storage;
  tf::cpp::cache<tf::cpp::default_index_t, float> cache;
  const tf::cpp::mesh<tf::cpp::default_index_t, float> value{
      storage.faces(), storage.points(), cache};
  return value.number_of_faces() == 0 ? 0 : 1;
}
