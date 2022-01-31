#pragma once

#include "device_convolution.cuh"

class OriginalConvolution
  : public IDeviceConvolution
{
  const int2 tileCount;

public:
  OriginalConvolution(int2 tileCount);

  void Compute(DeviceConvolutionParams& params) const override;
  std::string ConvolutionName() const override { return "OriginalConvolution"; }
  bool IsFused() const override { return false; }
};
