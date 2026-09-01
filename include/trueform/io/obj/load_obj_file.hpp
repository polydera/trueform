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

#include "../../core/buffer.hpp"

#include <cstddef>
#include <fstream>
#include <string>
#include <string_view>

namespace tf::io::obj {

inline auto load_obj_file(std::string_view path, tf::buffer<char> &output)
    -> bool {
  std::ifstream file(std::string(path), std::ios::binary | std::ios::ate);
  if (!file)
    return false;

  const auto size = file.tellg();
  if (size <= 0)
    return false;

  output.allocate(static_cast<std::size_t>(size));
  file.seekg(0, std::ios::beg);
  return static_cast<bool>(file.read(output.begin(), size));
}

} // namespace tf::io::obj
