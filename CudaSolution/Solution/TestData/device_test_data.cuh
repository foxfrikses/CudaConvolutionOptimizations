#pragma once

#include "test_data.cuh"

#include <thrust/device_vector.h>

struct DeviceTestData {
  const thrust::device_vector<float> input;
  const thrust::device_vector<float> dwFilter;
  const thrust::device_vector<float> dwBias;
  const thrust::device_vector<float> dFilter;
  const thrust::device_vector<float> dBias;
  thrust::device_vector<float> interimOutput;
  thrust::device_vector<float> output;

  explicit DeviceTestData(const TestData&);
  void ClearOutputs();
};
