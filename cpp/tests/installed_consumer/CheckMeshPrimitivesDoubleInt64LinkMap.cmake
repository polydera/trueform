if(NOT DEFINED LINK_MAP OR LINK_MAP STREQUAL "")
  message(FATAL_ERROR "LINK_MAP is required")
endif()
if(NOT EXISTS "${LINK_MAP}")
  message(FATAL_ERROR "typed mesh-primitives link map does not exist: ${LINK_MAP}")
endif()

file(READ "${LINK_MAP}" link_map)

set(expected_object "mesh_primitives_double_int64_3d.cpp.o")
string(FIND "${link_map}" "${expected_object}" expected_position)
if(expected_position EQUAL -1)
  message(FATAL_ERROR
    "typed mesh-primitives link omitted its requested shard: ${expected_object}")
endif()

foreach(unrelated_object IN ITEMS
    mesh_primitives_float_int32_3d.cpp.o
    mesh_primitives_float_int64_3d.cpp.o
    mesh_primitives_double_int32_3d.cpp.o)
  string(FIND "${link_map}" "${unrelated_object}" unrelated_position)
  if(NOT unrelated_position EQUAL -1)
    message(FATAL_ERROR
      "typed mesh-primitives link extracted unrelated object: ${unrelated_object}")
  endif()
endforeach()

message(STATUS
  "Double/int64/3D mesh-primitives link extracted only its requested production shard")
