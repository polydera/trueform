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

#include "file/mapped_file.hpp"
#include "obj/obj_execution_tuning.hpp"
#include "obj/read_complete_obj.hpp"
#include "obj/read_dynamic_obj.hpp"
#include "obj/read_fixed_obj.hpp"
#include "obj/read_fixed_obj_parallel.hpp"

#include "../core/blocked_buffer.hpp"
#include "../core/buffer.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/points_buffer.hpp"
#include "../core/range.hpp"
#include "../core/static_size.hpp"
#include "obj_file.hpp"

#include <tbb/task_arena.h>

#include <cstddef>
#include <string_view>

namespace tf::io {

/// @ingroup io
/// @brief Reader for position-only and complete ASCII OBJ data.
class obj_reader {
public:
  /// @pre The file is not modified for the duration of the read.
  template <typename Index, typename RealT, std::size_t Dims>
  auto read(std::string_view path, tf::points_buffer<RealT, Dims> &points,
            tf::offset_block_buffer<Index, Index> &faces) -> bool {
    tf::io::mapped_file file(path);
    if (!file)
      return false;
    return read(tf::make_range(file.data(), file.size()), points, faces);
  }

  template <typename Index, typename RealT, std::size_t Dims>
  auto read(tf::range<const char *, tf::dynamic_size> data,
            tf::points_buffer<RealT, Dims> &points,
            tf::offset_block_buffer<Index, Index> &faces) -> bool {
    return obj::read_dynamic_obj(data, points, faces);
  }

  /// @pre The file is not modified for the duration of the read.
  template <typename Index, typename RealT, std::size_t Dims, std::size_t Ngon>
  auto read(std::string_view path, tf::points_buffer<RealT, Dims> &points,
            tf::blocked_buffer<Index, Ngon> &faces) -> bool {
    tf::io::mapped_file file(path);
    if (!file)
      return false;
    return read(tf::make_range(file.data(), file.size()), points, faces);
  }

  template <typename Index, typename RealT, std::size_t Dims, std::size_t Ngon>
  auto read(tf::range<const char *, tf::dynamic_size> data,
            tf::points_buffer<RealT, Dims> &points,
            tf::blocked_buffer<Index, Ngon> &faces) -> bool {
    if (data.size() < obj::obj_execution_tuning::parallel_file_bytes ||
        tbb::this_task_arena::max_concurrency() < 2)
      return obj::read_fixed_obj(data, points, faces);
    return obj::read_fixed_obj_parallel(data, points, faces);
  }

  /// @pre The file is not modified for the duration of the read.
  template <typename Index, typename RealT, std::size_t Ngon>
  auto read(std::string_view path, tf::obj_file<Index, RealT, Ngon> &output)
      -> bool {
    tf::io::mapped_file file(path);
    if (!file)
      return false;
    return read(tf::make_range(file.data(), file.size()), output);
  }

  template <typename Index, typename RealT, std::size_t Ngon>
  auto read(tf::range<const char *, tf::dynamic_size> data,
            tf::obj_file<Index, RealT, Ngon> &output) -> bool {
    return obj::read_complete_obj(data, output);
  }
};

} // namespace tf::io
