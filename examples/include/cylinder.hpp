#pragma once

#include <trueform/core.hpp>

namespace examples {

/// Generate an open tube (cylinder without caps) as a quad mesh.
/// @param radius Radius of the tube
/// @param height Height of the tube
/// @param segments Number of segments around the circumference
/// @return Polygons buffer containing the tube mesh
auto make_tube(float radius, float height, int segments)
    -> tf::polygons_buffer<int, float, 3, 4>;

/// Generate a disk (filled circle) as a triangle fan mesh.
/// @param radius Radius of the disk
/// @param z Z coordinate of the disk plane
/// @param segments Number of segments around the circumference
/// @return Polygons buffer containing the disk mesh
auto make_disk(float radius, float z, int segments)
    -> tf::polygons_buffer<int, float, 3, 3>;

} // namespace examples
