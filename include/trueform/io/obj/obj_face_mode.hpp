/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */
#pragma once

namespace tf::io::obj {

enum class obj_face_mode { unknown, v, v_vt, v_vn, v_vt_vn };

inline auto obj_face_mode_names_textures(obj_face_mode mode) -> bool {
  return mode == obj_face_mode::v_vt || mode == obj_face_mode::v_vt_vn;
}

inline auto obj_face_mode_names_normals(obj_face_mode mode) -> bool {
  return mode == obj_face_mode::v_vn || mode == obj_face_mode::v_vt_vn;
}

} // namespace tf::io::obj
