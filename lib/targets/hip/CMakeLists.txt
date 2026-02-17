# #########################################################################################################################
# Additional sources
target_sources(quda_cpp PRIVATE  quda_api.cpp device.cpp malloc.cpp blas_lapack_hipblas.cpp comm_target.cpp)

# Keep HIP backend host-side translation units as C++ to avoid
# running HIP module pass instrumentation on non-kernel files.
set_source_files_properties(
	quda_api.cpp
	device.cpp
	malloc.cpp
	blas_lapack_hipblas.cpp
	comm_target.cpp
	PROPERTIES LANGUAGE CXX)

