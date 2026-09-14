if(NOT DEFINED LINK_MAP OR LINK_MAP STREQUAL "")
  message(FATAL_ERROR "LINK_MAP is required")
endif()
if(NOT EXISTS "${LINK_MAP}")
  message(FATAL_ERROR "double/int64 CSG graph link map does not exist: ${LINK_MAP}")
endif()

file(READ "${LINK_MAP}" link_map)

set(unrelated_link_map "${link_map}")
foreach(expected_object
    "csg_graph_double_int64.cpp.o"
    "graph_builders_double_int64_int64_3d.cpp.o")
  string(FIND "${link_map}" "${expected_object}" expected_position)
  if(expected_position EQUAL -1)
    message(FATAL_ERROR
      "double/int64 CSG graph link omitted its requested shard: ${expected_object}")
  endif()
  string(REPLACE "${expected_object}" "" unrelated_link_map
    "${unrelated_link_map}")
endforeach()
if(unrelated_link_map MATCHES "csg_graph_(float|double)(|_int64)\\.cpp\\.o")
  message(FATAL_ERROR
    "double/int64 CSG graph link extracted a legacy or sibling CSG graph shard")
endif()
if(unrelated_link_map MATCHES "graph_builders_(float|double)_int(32|64)_int(32|64)_3d\\.cpp\\.o")
  message(FATAL_ERROR
    "double/int64 CSG graph link extracted a sibling graph builder shard")
endif()

message(STATUS
  "Double/int64 CSG graph and repeated consumers extracted only their typed graph shard")
