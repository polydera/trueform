/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#include "trueform/python/io.hpp"
#include "trueform/python/io/read_stl.hpp"

namespace tf::py {

auto register_io(nanobind::module_ &m) -> void {
  // Register IO components
  register_io_read_stl(m);
}

} // namespace tf::py
