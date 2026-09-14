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
#include "trueform/cpp/topology/domain_labels.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace tf::cpp {
namespace {

/// A domain label is a PAIR, so the empty one states the two columns it
/// would have had.
template <typename Index>
auto empty_domain_label_pairs() -> nd_array<Index> {
  tf::buffer<Index> labels;
  labels.allocate(0);
  return nd_array<Index>::from_buffer(std::move(labels), {0, 2});
}

} // namespace

template <typename Index>
domain_labels_result<Index>::domain_labels_result()
    : _labels(empty_domain_label_pairs<Index>()) {}

template <typename Index>
domain_labels_result<Index>::domain_labels_result(nd_array<Index> labels,
                                                  Index number_of_domains,
                                                  Index outer_shell_label)
    : _labels(std::move(labels)), _number_of_domains(number_of_domains),
      _outer_shell_label(outer_shell_label) {
  if (!_labels.is_valid() || _labels.ndim() != 2 || _labels.shape_at(1) != 2)
    throw std::invalid_argument(
        "domain_labels_result: labels must have shape [N, 2]");
  if (_number_of_domains < Index{0})
    throw std::invalid_argument(
        "domain_labels_result: number of domains must be nonnegative");
  if (_outer_shell_label < Index{-1} || _outer_shell_label > _number_of_domains)
    throw std::out_of_range(
        "domain_labels_result: outer shell label out of range");
  for (const auto value : _labels)
    if (value < Index{0} || value > sentinel_label())
      throw std::out_of_range("domain_labels_result: label out of range");
}

template <typename Index>
auto domain_labels_result<Index>::labels() const -> nd_array<Index> {
  return _labels;
}

template <typename Index>
auto domain_labels_result<Index>::number_of_faces() const -> Index {
  return static_cast<Index>(_labels.shape_at(0));
}

template <typename Index>
auto domain_labels_result<Index>::number_of_domains() const -> Index {
  return _number_of_domains;
}

template <typename Index>
auto domain_labels_result<Index>::valid_label_begin() const -> Index {
  return Index{0};
}

template <typename Index>
auto domain_labels_result<Index>::valid_label_end() const -> Index {
  return _number_of_domains;
}

template <typename Index>
auto domain_labels_result<Index>::sentinel_label() const -> Index {
  return _number_of_domains;
}

template <typename Index>
auto domain_labels_result<Index>::outer_shell_label() const -> Index {
  return _outer_shell_label;
}

template <typename Index>
auto domain_labels_result<Index>::has_outer_shell_domain() const -> bool {
  return _outer_shell_label >= valid_label_begin() &&
         _outer_shell_label < valid_label_end();
}

template <typename Index>
auto domain_labels_result<Index>::empty() const -> bool {
  return number_of_faces() == Index{0};
}

template <typename Index>
auto domain_labels_result<Index>::label(Index face, Index side) const -> Index {
  if (face < Index{0} || face >= number_of_faces())
    throw std::out_of_range("domain_labels_result: face index out of range");
  if (side < Index{0} || side > Index{1})
    throw std::out_of_range("domain_labels_result: side index out of range");
  return _labels[static_cast<std::size_t>(face) * 2 +
                 static_cast<std::size_t>(side)];
}

#define TF_CPP_INSTANTIATE_DOMAIN_LABELS_RESULT(Index)                         \
  template class domain_labels_result<Index>

TF_CPP_MATRIX_FOR_EACH_INDEX(TF_CPP_INSTANTIATE_DOMAIN_LABELS_RESULT)

#undef TF_CPP_INSTANTIATE_DOMAIN_LABELS_RESULT

} // namespace tf::cpp
