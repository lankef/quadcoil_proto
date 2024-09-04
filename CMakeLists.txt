cmake_minimum_required(VERSION 3.14)
project(biest_call LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable profiling
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")

# Find Python and NumPy
set(Python_FIND_STRATEGY LOCATION)
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)

# Find pybind11 using Python
execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH ${pybind11_DIR})

# Add xtensor and xtensor-python as subdirectories because they
# have their own CMakeLists.txt
add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/xtl/)
add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/xtensor/)
add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/xtensor-python/)

# Adding BIEST and SCTL as part of the current CMakeLists.txt
include_directories(${CMAKE_SOURCE_DIR}/thirdparty/SCTL/include/)
include_directories(${CMAKE_SOURCE_DIR}/thirdparty/BIEST/include/)

# SCTL uses 
set(SCTL_HAVE_BLAS ON)
set(SCTL_HAVE_LAPACK ON)
set(SCTL_HAVE_LIBMVEC ON)
set(SCTL_HAVE_MPI ON)
set(SCTL_HAVE_FFTW ON)
# set(SCTL_HAVE_PVFMM ON)

# Find packages
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

# Define your extension
# add_library(biest_call MODULE src/biest_call_cpp/biest_call.cpp)
pybind11_add_module(biest_call src/biest_call_cpp/biest_call.cpp)
target_link_libraries(biest_call PRIVATE OpenMP::OpenMP_CXX xtensor xtensor-python pybind11::module)
target_include_directories(biest_call PRIVATE ${Python_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
