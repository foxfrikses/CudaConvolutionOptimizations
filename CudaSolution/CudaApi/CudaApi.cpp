#include "CudaApi.h"

#include <cuda_profiler_api.h>

namespace Cuda {
  void ProfilerStart() 
  {
    if (cudaSuccess != cudaProfilerStart()) {
      throw std::runtime_error("cudaProfilerStart failed!");
    }
  }

  void ProfilerStop()
  {
    if (cudaSuccess != cudaProfilerStop()) {
      throw std::runtime_error("cudaProfilerStart failed!");
    }
  }

  void SetDevice(int device)
  {
    if (cudaSuccess != cudaSetDevice(device)) {
      throw std::runtime_error("cudaSetDevice failed! "
                               "Do you have a CUDA-capable GPU installed?");
    }
  }

  void ResetDevice() {
    auto cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
      throw std::runtime_error("cudaDeviceReset failed!");
    }
  }

  void DeviceSynchronize()
  {
    auto cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
      throw std::runtime_error("cudaDeviceSynchronize returned error code " +
                               std::to_string(cudaStatus));
    }
  }

  void CheckForLastError() {
    auto cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(cudaStatus));
    }
  }
}
