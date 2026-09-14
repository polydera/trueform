if(NOT DEFINED LINK_MAP OR LINK_MAP STREQUAL "")
  message(FATAL_ERROR "LINK_MAP is required")
endif()
if(NOT EXISTS "${LINK_MAP}")
  message(FATAL_ERROR "registration link map does not exist: ${LINK_MAP}")
endif()

file(READ "${LINK_MAP}" link_map)

string(FIND "${link_map}" "registration_fit_float_2d.cpp.o" float_position)
if(float_position EQUAL -1)
  message(FATAL_ERROR
    "float 2D registration link omitted its explicit-instantiation shard")
endif()

if(link_map MATCHES "registration_fit_double_2d\\.cpp\\.o")
  message(FATAL_ERROR
    "float 2D registration link extracted the unrelated double shard")
endif()
if(link_map MATCHES "registration_fit_(float|double)_3d\\.cpp\\.o")
  message(FATAL_ERROR
    "float 2D registration link extracted an unrelated 3D shard")
endif()

message(STATUS
  "Float 2D registration link extracted only its requested axis shard")
