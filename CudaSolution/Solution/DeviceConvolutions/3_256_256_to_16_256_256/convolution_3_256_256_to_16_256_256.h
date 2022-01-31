#pragma once

#include "device_convolution.cuh"

class Convolution_3_256_256_to_16_256_256 
  : public IDeviceConvolution
{
public:
  void Compute(DeviceConvolutionParams& params) const override;
  std::string ConvolutionName() const override { return "Convolution_3_256_256_to_16_256_256"; }
  bool IsFused() const override { return false; }
};
