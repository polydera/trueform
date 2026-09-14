if(NOT DEFINED LINK_MAP OR LINK_MAP STREQUAL "")
  message(FATAL_ERROR "LINK_MAP is required")
endif()
if(NOT EXISTS "${LINK_MAP}")
  message(FATAL_ERROR
    "float/int64/2D polygon-soup cleaning link map does not exist: ${LINK_MAP}")
endif()

file(READ "${LINK_MAP}" link_map)

set(expected_object "clean_soups_float_int64_2d.cpp.o")
string(FIND "${link_map}" "${expected_object}" expected_position)
if(expected_position EQUAL -1)
  message(FATAL_ERROR
    "float/int64/2D polygon-soup cleaning link omitted its requested shard")
endif()

string(REPLACE "${expected_object}" "" unrelated_link_map "${link_map}")
if(unrelated_link_map MATCHES "clean_(points_(float|double)_(2d|3d)|forms_(float|double)_int(32|64)_(2d|3d)|soups_(float|double)_int(32|64)_(2d|3d))\\.cpp\\.o")
  message(FATAL_ERROR
    "float/int64/2D polygon-soup cleaning link extracted a sibling cleaning shard")
endif()

message(STATUS
  "Float/int64/2D segment and triangle soup cleaning extracted only its soup shard")
