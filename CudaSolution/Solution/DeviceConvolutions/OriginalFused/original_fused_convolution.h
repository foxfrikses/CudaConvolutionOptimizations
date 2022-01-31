#pragma once

#include "device_convolution.cuh"

class OriginalFusedConvolution
  : public IDeviceConvolution
{
  const int2 tileCount;

public:
  OriginalFusedConvolution(int2 tileCount);

  void Compute(DeviceConvolutionParams& params) const override;
  std::string ConvolutionName() const override { return "OriginalFusedConvolution"; }
  bool IsFused() const override { return true; }
};
#
