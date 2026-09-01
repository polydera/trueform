/*
 * Copyright (c) 2026 XLAB
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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/views/blocked_range.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Per-domain operand-inclusion bitvectors.
///
/// Flat `uint32_t` buffer with one block of `words_per_domain` words
/// per domain. Access per-domain bitvectors via @ref make_range, which
/// returns a dynamic-block-size view: `range[d]` yields the
/// `words_per_domain` word span for domain `d`. Where a single form's bit
/// is the question, @ref set and @ref test own where that bit sits.
struct domain_inclusions {
  tf::buffer<std::uint32_t> bits;
  std::size_t words_per_domain = 0;

  /// @brief View as a blocked range over `bits`, one block per domain.
  auto make_range() const {
    return tf::make_blocked_range(tf::make_range(bits), words_per_domain);
  }

  /// @brief Set form `tag`'s bit in domain `d`'s block.
  auto set(std::size_t d, std::size_t tag) -> void {
    bits[d * words_per_domain + tag / 32u] |= std::uint32_t(1)
                                              << (tag % 32u);
  }

  /// @brief Clear form `tag`'s bit in domain `d`'s block.
  auto clear(std::size_t d, std::size_t tag) -> void {
    bits[d * words_per_domain + tag / 32u] &=
        ~(std::uint32_t(1) << (tag % 32u));
  }

  /// @brief Whether form `tag`'s bit is set in domain `d`'s block.
  auto test(std::size_t d, std::size_t tag) const -> bool {
    return (bits[d * words_per_domain + tag / 32u] &
            (std::uint32_t(1) << (tag % 32u))) != 0u;
  }
};

/// @ingroup csg_graph_internals
/// @brief The inclusion table of an N-form arrangement, in the state every
///        bit is stated from: outside everything.
///
/// Bit `inclusion[d * words_per_domain + i/32] & (1 << (i % 32))` is set
/// iff domain `d` is inside form `i`. The seeds are cast by
/// @ref tf::csg::graph::seed_inclusion_bits and every other domain is
/// reached from them by @ref tf::csg::graph::propagate_inclusion_bits,
/// which is the sole authority for what a domain is inside of.
///
/// @param n_tags    The number of forms in the arrangement.
/// @param n_domains The arrangement descriptor's domain count.
template <typename Index>
auto make_domain_inclusions(Index n_tags, Index n_domains)
    -> domain_inclusions {
  domain_inclusions out;
  out.words_per_domain = static_cast<std::size_t>((n_tags + 31) / 32);
  out.bits.allocate(static_cast<std::size_t>(n_domains) *
                    out.words_per_domain);
  tf::parallel_fill(out.bits, std::uint32_t(0));
  return out;
}

} // namespace tf::csg::graph
