#ifndef HIP_CHECK_ERROR
#define HIP_CHECK_ERROR

void inline checkError(hipError_t hipErr, const char * functWithError)
{
  if ( hipErr != hipSuccess )
  {
    fprintf(stderr, "ERROR %s - %s\n", functWithError,
            hipGetErrorString(hipErr));
    exit(-1);
  }
}

#endif
