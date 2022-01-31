#pragma once

#include "device_convolution.cuh"

class SizeFlexibleOptimizedConvolution
  : public IDeviceConvolution
{
public:
  void Compute(DeviceConvolutionParams& params) const override;
  std::string ConvolutionName() const override { return "SizeFlexibleOptimizedConvolution"; }
  bool IsFused() const override { return false; }
};
