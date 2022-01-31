#include "device_test_data.cuh"

#include <thrust/fill.h>

DeviceTestData::DeviceTestData(const TestData& td)
  : input(td.input)
  , dwFilter(td.dwFilter)
  , dwBias(td.dwBias)
  , dFilter(td.dFilter)
  , dBias(td.dBias)
  , interimOutput(td.interimOutput.size())
  , output(td.output.size())
{}

void DeviceTestData::ClearOutputs()
{
  thrust::fill(interimOutput.begin(), interimOutput.end(), 0.0f);
  thrust::fill(output.begin(), output.end(), 0.0f);
}
