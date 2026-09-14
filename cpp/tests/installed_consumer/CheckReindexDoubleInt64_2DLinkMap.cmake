if(NOT DEFINED LINK_MAP OR LINK_MAP STREQUAL "")
  message(FATAL_ERROR "LINK_MAP is required")
endif()
if(NOT EXISTS "${LINK_MAP}")
  message(FATAL_ERROR
    "double/int64/2D typed reindex link map does not exist: ${LINK_MAP}")
endif()

file(READ "${LINK_MAP}" link_map)

set(expected_objects
  "split_components_double_int64_2d.cpp.o"
  "split_components_float_int32_2d.cpp.o")
foreach(expected_object IN LISTS expected_objects)
  string(FIND "${link_map}" "${expected_object}" expected_position)
  if(expected_position EQUAL -1)
    message(FATAL_ERROR
      "double/int64/2D typed reindex link omitted ${expected_object}")
  endif()
  string(REPLACE "${expected_object}" "" link_map "${link_map}")
endforeach()

if(link_map MATCHES "reindex\\.cpp\\.o|split_components_(float|double)_int(32|64)_(2d|3d)\\.cpp\\.o")
  message(FATAL_ERROR
    "double/int64/2D typed reindex link extracted the legacy or an unrelated typed shard")
endif()

message(STATUS
  "Typed V3/dynamic reindex, mixed concatenation, and domain split extracted exactly the double/int64/2D and reverse float/int32/2D shards")
