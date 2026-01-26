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

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <string>
#include <trueform/core/range.hpp>
#include <trueform/core/views/offset_block_range.hpp>

namespace tf::py {

template <typename IndexT, typename ValueT> class offset_blocked_array_wrapper {
public:
  offset_blocked_array_wrapper() = default;

  offset_blocked_array_wrapper(
      nanobind::ndarray<nanobind::numpy, IndexT, nanobind::shape<-1>>
          offsets_array,
      nanobind::ndarray<nanobind::numpy, ValueT, nanobind::shape<-1>>
          data_array)
      : _offsets_array{offsets_array}, _data_array{data_array} {
    // Validate
    if (offsets_array.size() > 0) {
      IndexT first = static_cast<IndexT *>(offsets_array.data())[0];
      IndexT last =
          static_cast<IndexT *>(offsets_array.data())[offsets_array.size() - 1];

      if (first != 0) {
        throw std::invalid_argument("First offset must be 0, got " +
                                    std::to_string(first));
      }
      if (last != static_cast<IndexT>(data_array.size())) {
        throw std::invalid_argument("Last offset must equal data size. Got " +
                                    std::to_string(last) + ", expected " +
                                    std::to_string(data_array.size()));
      }
    }
  }

  auto
  set_arrays(nanobind::ndarray<nanobind::numpy, IndexT, nanobind::shape<-1>>
                 offsets_array,
             nanobind::ndarray<nanobind::numpy, ValueT, nanobind::shape<-1>>
                 data_array) -> void {
    _offsets_array = std::move(offsets_array);
    _data_array = std::move(data_array);
  }

  // Create view into Python-owned arrays (mutable)
  auto make_range() {
    IndexT *offsets_data = static_cast<IndexT *>(_offsets_array.data());
    ValueT *data_data = static_cast<ValueT *>(_data_array.data());

    auto offsets_range = tf::make_range(offsets_data, _offsets_array.size());
    auto data_range = tf::make_range(data_data, _data_array.size());

    return tf::make_offset_block_range(offsets_range, data_range);
  }

  // Create view into Python-owned arrays (const)
  auto make_range() const {
    const IndexT *offsets_data =
        static_cast<const IndexT *>(_offsets_array.data());
    const ValueT *data_data = static_cast<const ValueT *>(_data_array.data());

    auto offsets_range = tf::make_range(offsets_data, _offsets_array.size());
    auto data_range = tf::make_range(data_data, _data_array.size());

    return tf::make_offset_block_range(offsets_range, data_range);
  }

  auto size() const -> std::size_t {
    if (_offsets_array.size() == 0) {
      return 0;
    }
    return _offsets_array.size() - 1;
  }

  auto offsets_array() const
      -> nanobind::ndarray<nanobind::numpy, IndexT, nanobind::shape<-1>> {
    return _offsets_array;
  }

  auto data_array() const
      -> nanobind::ndarray<nanobind::numpy, ValueT, nanobind::shape<-1>> {
    return _data_array;
  }

private:
  nanobind::ndarray<nanobind::numpy, IndexT, nanobind::shape<-1>>
      _offsets_array;
  nanobind::ndarray<nanobind::numpy, ValueT, nanobind::shape<-1>> _data_array;
};

// Forward declaration of registration function
auto register_offset_blocked_array(nanobind::module_ &m) -> void;

} // namespace tf::py
