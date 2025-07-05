/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

namespace tf {
enum class intersection_type : char {
  vertex_vertex = 0,
  vertex_edge = 1,
  edge_vertex = 2,
  edge_edge = 3,
  vertex_face = 4,
  face_vertex = 5,
  edge_face = 6,
  face_edge = 7
};
}
