#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include <thrust/device_ptr.h>

namespace Cuda {
  void ProfilerStart();
  void ProfilerStop();

  void SetDevice(int device = 0);
  void ResetDevice();

  void DeviceSynchronize();
  void CheckForLastError();
 }