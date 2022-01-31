#pragma once

#include <vector>
#include <memory>

#include "test_data.cuh"
#include "device_test_data.cuh"
#include "device_convolution.cuh"

class DeviceConvolutionTester final {
  TestSize testSize;

  std::vector<std::shared_ptr<IDeviceConvolution>> convolutions;

public:
  void SetTestSize(const TestSize&);

  void AddConvolution(std::shared_ptr<IDeviceConvolution>);
  void ResetConvolutions();

  void CheckForAccuracy(int testCount);
  void Profile(int testCount);
};
