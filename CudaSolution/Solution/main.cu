#include <iostream>

#include "CudaApi.h"

#include "convolution_3_256_256_to_16_256_256.h"
#include "size_flexible_optimized_convolution.h"
#include "original_convolution.h"
#include "original_fused_convolution.h"

#include "device_convolution_tester.h"

int main()
{
  Cuda::SetDevice();

  constexpr bool
    needProfile = true,
    needCheckAccuracy = true,
    needRunFirstTestSet = true,
    needRunSecondTestSet = true,
    needRunThirdTestSet = true;

  constexpr int
    accuracyTestCount = 10,
    profileTestCount = 10;

  if (needRunFirstTestSet) {
    DeviceConvolutionTester tester;
    tester.SetTestSize([] {
      TestSize s;
      s.nInputChannel = 3;
      s.nOutputChannel = 16;
      s.shape = {256, 256};
      return s;
    }());
    tester.AddConvolution(std::make_unique<OriginalConvolution>(int2{8, 8}));
    tester.AddConvolution(std::make_unique<OriginalFusedConvolution>(int2{8, 8}));
    tester.AddConvolution(std::make_unique<SizeFlexibleOptimizedConvolution>());
    tester.AddConvolution(std::make_unique<Convolution_3_256_256_to_16_256_256>());
    if (needCheckAccuracy) tester.CheckForAccuracy(accuracyTestCount);
    if (needProfile) tester.Profile(profileTestCount);
  }

  if (needRunSecondTestSet) {
    DeviceConvolutionTester tester;
    tester.SetTestSize([] {
      TestSize s;
      s.nInputChannel = 32;
      s.nOutputChannel = 64;
      s.shape = {64, 64};
      return s;
    }());
    tester.AddConvolution(std::make_unique<OriginalConvolution>(int2{8, 8}));
    tester.AddConvolution(std::make_unique<OriginalFusedConvolution>(int2{8, 8}));
    tester.AddConvolution(std::make_unique<SizeFlexibleOptimizedConvolution>());
    if (needCheckAccuracy) tester.CheckForAccuracy(accuracyTestCount);
    if (needProfile) tester.Profile(profileTestCount);
  }

  if (needRunThirdTestSet) {
    DeviceConvolutionTester tester;
    tester.SetTestSize([] {
      TestSize s;
      s.nInputChannel = 256;
      s.nOutputChannel = 256;
      s.shape = {16, 16};
      return s;
    }());
    tester.AddConvolution(std::make_unique<OriginalConvolution>(int2{4, 4}));
    tester.AddConvolution(std::make_unique<OriginalFusedConvolution>(int2{4, 4}));
    tester.AddConvolution(std::make_unique<SizeFlexibleOptimizedConvolution>());
    if (needCheckAccuracy) tester.CheckForAccuracy(accuracyTestCount);
    if (needProfile) tester.Profile(profileTestCount);
  }

  Cuda::ResetDevice();

  return 0;
}
