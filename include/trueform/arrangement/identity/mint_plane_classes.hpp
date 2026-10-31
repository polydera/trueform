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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/config/verify.hpp"
#include "../../core/views/sequence_range.hpp"

#include <cstddef>

namespace tf::arrangement {

/// A wave's births joined to the plane layer's classes: `delta_class_t` and
/// `class_name_of(cls)` state the wave's classes, `delta_created_class` per
/// identity it mints the class that identity names. A class is read only
/// through the identity it minted, so only a minting class persists, once,
/// in class order past `class_t`; every minted identity names its class
/// there.
template <typename Index, typename Param, typename ClassName, typename Name>
auto mint_plane_classes(const tf::buffer<Param> &delta_class_t,
                        const ClassName &class_name_of,
                        const tf::buffer<Index> &delta_created_class,
                        tf::buffer<Param> &class_t,
                        tf::buffer<Name> &class_name,
                        tf::buffer<Index> &created_class) -> void {
  const auto n_delta_classes = delta_class_t.size();
  const auto class_base = Index(class_t.size());
  if constexpr (tf::core::verification)
    for (const auto cls : delta_created_class)
      tf::core::verify(cls >= Index(0) && std::size_t(cls) < n_delta_classes,
                       "a minted identity names a class the wave stated");
  // one coincidence mints once, so each class is marked once
  tf::buffer<Index> class_slot;
  class_slot.allocate_and_initialize(n_delta_classes + 1, Index(0));
  tf::parallel_for_each(
      delta_created_class,
      [&class_slot](Index cls) { class_slot[std::size_t(cls) + 1] = Index(1); },
      tf::checked);
  for (std::size_t cls = 1; cls < class_slot.size(); ++cls)
    class_slot[cls] += class_slot[cls - 1];
  const auto n_kept_classes = std::size_t(class_slot[class_slot.size() - 1]);
  class_t.reallocate(std::size_t(class_base) + n_kept_classes);
  class_name.reallocate(std::size_t(class_base) + n_kept_classes);
  tf::parallel_for_each(
      tf::make_sequence_range(n_delta_classes),
      [&](std::size_t cls) {
        if (class_slot[cls + 1] == class_slot[cls])
          return;
        const auto slot =
            std::size_t(class_base) + std::size_t(class_slot[cls]);
        class_t[slot] = delta_class_t[cls];
        class_name[slot] = class_name_of(cls);
      },
      tf::checked);
  const auto created_write = created_class.size();
  created_class.reallocate(created_write + delta_created_class.size());
  tf::parallel_for_each(
      tf::make_sequence_range(delta_created_class.size()),
      [&](std::size_t at) {
        created_class[created_write + at] =
            class_slot[std::size_t(delta_created_class[at])] + class_base;
      },
      tf::checked);
}

} // namespace tf::arrangement
