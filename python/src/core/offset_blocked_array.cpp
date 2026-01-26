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

#include "trueform/python/core/offset_blocked_array.hpp"

namespace tf::py {

auto register_offset_blocked_array(nanobind::module_ &m) -> void {
  // Register OffsetBlockRange for int32, int32
  nanobind::class_<offset_blocked_array_wrapper<int, int>>(
      m, "OffsetBlockedArrayWrapperIntInt")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int, nanobind::shape<-1>>,
           nanobind::ndarray<nanobind::numpy, int, nanobind::shape<-1>>>())
      .def("size", &offset_blocked_array_wrapper<int, int>::size)
      .def("offsets_array",
           &offset_blocked_array_wrapper<int, int>::offsets_array)
      .def("data_array", &offset_blocked_array_wrapper<int, int>::data_array);

  // Register OffsetBlockRange for int64, int64
  nanobind::class_<offset_blocked_array_wrapper<int64_t, int64_t>>(
      m, "OffsetBlockedArrayWrapperInt64Int64")
      .def(nanobind::init<
           nanobind::ndarray<nanobind::numpy, int64_t, nanobind::shape<-1>>,
           nanobind::ndarray<nanobind::numpy, int64_t, nanobind::shape<-1>>>())
      .def("size", &offset_blocked_array_wrapper<int64_t, int64_t>::size)
      .def("offsets_array",
           &offset_blocked_array_wrapper<int64_t, int64_t>::offsets_array)
      .def("data_array",
           &offset_blocked_array_wrapper<int64_t, int64_t>::data_array);
}

// Explicit template instantiations
template class offset_blocked_array_wrapper<int, int>;
template class offset_blocked_array_wrapper<int64_t, int64_t>;

} // namespace tf::py
