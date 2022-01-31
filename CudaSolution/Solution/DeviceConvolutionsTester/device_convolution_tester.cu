#include "device_convolution_tester.h"

#include "utils.h"

#include "CudaApi.h"

#include <thrust/copy.h>

#include <unordered_map>
#include <iostream>

void DeviceConvolutionTester::SetTestSize(
  const TestSize& size
)
{
  testSize = size;
}

void DeviceConvolutionTester::AddConvolution(
  std::shared_ptr<IDeviceConvolution> convolution
)
{
  convolutions.push_back(std::move(convolution));
}

void DeviceConvolutionTester::ResetConvolutions()
{
  convolutions.clear();
}

void DeviceConvolutionTester::CheckForAccuracy(
  int testCount
)
{
  if (testCount <= 0 || convolutions.empty()) {
    return;
  }

  struct ErrorCount {int output = 0, interimOutput = 0;};
  std::unordered_map<std::shared_ptr<IDeviceConvolution>, ErrorCount> convErrors;

  for (int test = 0; test < testCount; ++test) {
    auto hostTestData = TestData::Generate(testSize);
    DeviceTestData deviceTestData(*hostTestData);

    DeviceConvolutionParams params;
    params.input          = deviceTestData.input.data();
    params.dwFilter       = deviceTestData.dwFilter.data();
    params.dwBias         = deviceTestData.dwBias.data();
    params.interimOutput  = deviceTestData.interimOutput.data();
    params.dFilter        = deviceTestData.dFilter.data();
    params.dBias          = deviceTestData.dBias.data();
    params.output         = deviceTestData.output.data();
    params.nInputChannel  = testSize.nInputChannel;
    params.nOutputChannel = testSize.nOutputChannel;
    params.shape          = testSize.shape;

    std::vector<float> interimOutput(hostTestData->interimOutput.size());
    std::vector<float> output(hostTestData->output.size());

    for (auto conv : convolutions) {
      deviceTestData.ClearOutputs();
      conv->Compute(params);
      Cuda::DeviceSynchronize();

      thrust::copy(deviceTestData.interimOutput.begin(),
                   deviceTestData.interimOutput.end(),
                   interimOutput.begin());
      thrust::copy(deviceTestData.output.begin(),
                   deviceTestData.output.end(),
                   output.begin());

      auto& errors = convErrors[conv];
      if (!conv->IsFused()) {
        errors.interimOutput += (AreEqual(interimOutput, hostTestData->interimOutput) ? 0 : 1);
      }
      errors.output += (AreEqual(output, hostTestData->output) ? 0 : 1);
    }
  }

  std::cout << "\n" << "---- INPUT " << testSize.nInputChannel <<
    "x(" <<testSize.shape.x << "," <<
    testSize.shape.y << ")";
  std::cout << "\n" << "---- OUTPUT " << testSize.nOutputChannel <<
    "x(" << testSize.shape.x << "," <<
    testSize.shape.y << ")";
  std::cout << std::endl;

  for (auto& convError : convErrors) {
    std::cout << "\n  " << convError.first->ConvolutionName() << '\n';
    
    if (convError.second.interimOutput == 0 && convError.second.output == 0) {
      std::cout << "All " << testCount << " tests have been passed" << std::endl;
    }
    else {
      if (!convError.first->IsFused()) {
        std::cout << "There are " << convError.second.interimOutput <<
          " from " << testCount << " incorrect interim outputs" << std::endl;
      }
      std::cout << "There are " << convError.second.output << 
        " from " << testCount << " incorrect outputs" << std::endl;
    }
  }
  
  std::cout << std::endl;
}

void DeviceConvolutionTester::Profile(
  int testCount
)
{
  auto hostTestData = TestData::Generate(testSize);
  DeviceTestData deviceTestData(*hostTestData);

  DeviceConvolutionParams params;
  params.input          = deviceTestData.input.data();
  params.dwFilter       = deviceTestData.dwFilter.data();
  params.dwBias         = deviceTestData.dwBias.data();
  params.interimOutput  = deviceTestData.interimOutput.data();
  params.dFilter        = deviceTestData.dFilter.data();
  params.dBias          = deviceTestData.dBias.data();
  params.output         = deviceTestData.output.data();
  params.nInputChannel  = testSize.nInputChannel;
  params.nOutputChannel = testSize.nOutputChannel;
  params.shape          = testSize.shape;

  Cuda::ProfilerStart();
  for (int test = 0; test < testCount; ++test) {
    for (auto conv : convolutions) {
      conv->Compute(params);
      Cuda::DeviceSynchronize();
    }
  }
  Cuda::ProfilerStop();
}
