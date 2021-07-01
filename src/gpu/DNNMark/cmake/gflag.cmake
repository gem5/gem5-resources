include(ExternalProject)

ExternalProject_Add(gflags
    GIT_REPOSITORY git clone https://github.com/gflags/gflags.git
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/gflags"
# Disable install step
    INSTALL_COMMAND ""
)

# Specify include dir
ExternalProject_Get_Property(gflags SOURCE_DIR)
set(GFLAGS_INCLUDE_DIRS ${SOURCE_DIR}/gflags/include)

# Specify MainTest's link libraries
ExternalProject_Get_Property(googletest binary_dir)
set(GFLAGS_LIBS_DIR ${binary_dir}/googlemock/gtest)
