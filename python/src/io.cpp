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

#include "trueform/python/io.hpp"
#include "trueform/python/io/read_obj.hpp"
#include "trueform/python/io/read_stl.hpp"
#include "trueform/python/io/write_obj.hpp"
#include "trueform/python/io/write_stl.hpp"

namespace tf::py {

auto register_io(nanobind::module_ &m) -> void {
  // Create io submodule
  auto io_module = m.def_submodule("io", "IO operations");

  // Register IO components on the submodule
  register_io_read_obj(io_module);
  register_io_read_stl(io_module);
  register_io_write_obj(io_module);
  register_io_write_stl(io_module);
}

} // namespace tf::py
