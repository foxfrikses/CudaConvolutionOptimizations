#pragma once

#include <thrust/device_ptr.h>

#include <vector_types.h>

#include <memory>
#include <string>

#include "device_test_data.cuh"
#include "test_data.cuh"

struct DeviceConvolutionParams {
  thrust::device_ptr<const float> input;         // [nInputChannel][shape.x][shape.y]
  thrust::device_ptr<const float> dwFilter;      // [nInputChannel][3][3]
  thrust::device_ptr<const float> dwBias;        // [nInputChannel]
  thrust::device_ptr<float>       interimOutput; // [nInputChannel][shape.x][shape.y]
  thrust::device_ptr<const float> dFilter;       // [nOutputChannel][nInputChannel][1][1]
  thrust::device_ptr<const float> dBias;         // [nOutputChannel]
  thrust::device_ptr<float>       output;        // [nOutputChannel][shape.x][shape.y]

  int nInputChannel;
  int nOutputChannel;

  int2 shape;
};

class IDeviceConvolution {
public:
  virtual void Compute(DeviceConvolutionParams& params) const = 0;
  virtual std::string ConvolutionName() const = 0;
  virtual bool IsFused() const { return false; };
};
