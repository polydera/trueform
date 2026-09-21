if(NOT DEFINED LINK_MAP OR LINK_MAP STREQUAL "")
  message(FATAL_ERROR "LINK_MAP is required")
endif()
if(NOT EXISTS "${LINK_MAP}")
  message(FATAL_ERROR "legacy carrier link map does not exist: ${LINK_MAP}")
endif()

file(READ "${LINK_MAP}" link_map)

# The carriers are the cache and the assembly: a cache is one translation unit
# for the whole matrix, as its structure always was, so what a legacy consumer
# extracts of them has no per-combination member to name.
set(expected_members
  primitive_float_3d.cpp.o
  primitive_double_3d.cpp.o
  distance_float_3d.cpp.o
  distance_double_3d.cpp.o)
foreach(expected_member IN LISTS expected_members)
  string(FIND "${link_map}" "${expected_member}" member_position)
  if(member_position EQUAL -1)
    message(FATAL_ERROR
      "legacy carrier link omitted expected archive member: ${expected_member}")
  endif()
endforeach()

set(forbidden_patterns
  "primitive_(float|double)_2d\\.cpp\\.o"
  "distance_(float|double|mixed)_(2d|int(32|64)_(2d|3d))\\.cpp\\.o"
  "distance_mixed_3d\\.cpp\\.o")
foreach(forbidden_pattern IN LISTS forbidden_patterns)
  if(link_map MATCHES "${forbidden_pattern}")
    message(FATAL_ERROR
      "legacy carrier link extracted a nonlegacy archive member matching: ${forbidden_pattern}")
  endif()
endforeach()

message(STATUS "Legacy carrier link extracted only the 3D shards its own reals name")
