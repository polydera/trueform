#pragma once
#ifndef TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#endif
#include "./tiny_obj_loader.h"
#include <string>
#include <vector>

namespace tf::examples {
inline auto read_mesh(const std::string &filename)
    -> std::pair<std::vector<float>, std::vector<int>> {
  tinyobj::ObjReader rd;
  rd.ParseFromFile(filename);
  auto &&pts = rd.GetAttrib().GetVertices();
  auto &&tris = rd.GetShapes()[0].mesh.indices;

  std::vector<int> triangles;

  triangles.resize(tris.size());
  auto *tptr = tris.data();
  for (int i = 0; i < int(triangles.size()); ++i) {
    triangles[i] = (tptr++)->vertex_index;
  }

  return std::make_pair(std::move(pts), std::move(triangles));
}
} // namespace tf::examples
