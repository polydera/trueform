if(NOT DEFINED LINK_MAP OR LINK_MAP STREQUAL "")
  message(FATAL_ERROR "LINK_MAP is required")
endif()
if(NOT EXISTS "${LINK_MAP}")
  message(FATAL_ERROR
    "legacy primitive-normal link map does not exist: ${LINK_MAP}")
endif()

file(READ "${LINK_MAP}" link_map)

foreach(expected_object IN ITEMS
    primitive_geometry_float_3d.cpp.o
    primitive_geometry_double_3d.cpp.o)
  string(FIND "${link_map}" "${expected_object}" expected_position)
  if(expected_position EQUAL -1)
    message(FATAL_ERROR
      "frozen primitive-normal ABI link omitted historical member: ${expected_object}")
  endif()
endforeach()

if(link_map MATCHES "normals(_(float|double)_int(32|64)_3d)?\\.cpp\\.o")
  message(FATAL_ERROR
    "frozen primitive-normal ABI link extracted a new mesh-normal shard")
endif()
if(link_map MATCHES "obj_(float|double)_int(32|64)_3d\\.cpp\\.o|stl_int(32|64)_3d\\.cpp\\.o|analysis_(float|double)_int(32|64)_(2d|3d)\\.cpp\\.o|topology_links_int(32|64)\\.cpp\\.o")
  message(FATAL_ERROR
    "frozen primitive-normal ABI link extracted an unrelated IO/topology shard")
endif()

message(STATUS
  "Frozen primitive-normal ABI resolved historical symbols without mesh-normal shards")
