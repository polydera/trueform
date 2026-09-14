if(NOT DEFINED LINK_MAP OR LINK_MAP STREQUAL "")
  message(FATAL_ERROR "LINK_MAP is required")
endif()
if(NOT EXISTS "${LINK_MAP}")
  message(FATAL_ERROR
    "double/int64/2D component-splitting link map does not exist: ${LINK_MAP}")
endif()

file(READ "${LINK_MAP}" link_map)

set(expected_object "split_components_double_int64_2d.cpp.o")
string(FIND "${link_map}" "${expected_object}" expected_position)
if(expected_position EQUAL -1)
  message(FATAL_ERROR
    "double/int64/2D component-splitting link omitted its requested shard")
endif()

string(REPLACE "${expected_object}" "" unrelated_link_map "${link_map}")
if(unrelated_link_map MATCHES "reindex\\.cpp\\.o|split_components_(float|double)_int(32|64)_(2d|3d)\\.cpp\\.o")
  message(FATAL_ERROR
    "double/int64/2D component-splitting link extracted a legacy or sibling shard")
endif()

message(STATUS
  "Double/int64/2D dynamic Mesh and EdgeMesh splitting extracted only their component shard")
