/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#include "trueform/python/io/read_stl.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <trueform/trueform.hpp>

namespace tf::py {

auto register_io_read_stl(nanobind::module_ &m) -> void {
  // Read STL file and return (faces, points) tuple as numpy arrays
  // Faces: (num_faces, 3) int array
  // Points: (num_points, 3) float array
  m.def("read_stl",
        [](const std::string &filename) {
          // Read STL file using C++ function
          auto polys = tf::read_stl<int>(filename);

          // Get sizes
          auto num_faces = polys.faces_buffer().size();
          auto num_points = polys.points_buffer().size();

          // Release ownership from C++ buffers
          int *faces_ptr = polys.faces_buffer().data_buffer().release();
          float *points_ptr = polys.points_buffer().data_buffer().release();

          // Create numpy arrays with ownership transfer
          // The arrays will manage the memory and delete it when they're done
          auto faces_capsule = nanobind::capsule(
              faces_ptr, [](void *p) noexcept { delete[] static_cast<int *>(p); });

          auto points_capsule = nanobind::capsule(points_ptr, [](void *p) noexcept {
            delete[] static_cast<float *>(p);
          });

          // Create numpy arrays from raw pointers
          // Shape: (num_faces, 3) for faces, (num_points, 3) for points
          auto faces = nanobind::ndarray<nanobind::numpy, int>(
              faces_ptr, {static_cast<size_t>(num_faces), 3}, faces_capsule);

          auto points = nanobind::ndarray<nanobind::numpy, float>(
              points_ptr, {num_points, 3}, points_capsule);

          // Return as tuple
          return nanobind::make_tuple(faces, points);
        },
        nanobind::arg("filename"),
        "Read an STL file and return (faces, points) tuple.\n\n"
        "Parameters\n"
        "----------\n"
        "filename : str\n"
        "    Path to the STL file\n\n"
        "Returns\n"
        "-------\n"
        "faces : ndarray of shape (num_faces, 3) and dtype int32\n"
        "    Face indices into the points array\n"
        "points : ndarray of shape (num_points, 3) and dtype float32\n"
        "    3D coordinates of mesh vertices");
}

} // namespace tf::py
