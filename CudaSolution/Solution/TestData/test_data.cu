#include "test_data.cuh"

#include "utils.h"

#include "convolution_cpu.h"

std::unique_ptr<TestData> TestData::Generate(const TestSize& s)
{
  vec input    = GenerateRandomNumArray(size_t(s.nInputChannel) * s.shape.x * s.shape.y);
  vec dwFilter = GenerateRandomNumArray(size_t(s.nInputChannel) * 3 * 3);
  vec dwBias   = GenerateRandomNumArray(size_t(s.nInputChannel), 0.0, 0.2);
  vec dFilter  = GenerateRandomNumArray(size_t(s.nOutputChannel) * s.nInputChannel);
  vec dBias    = GenerateRandomNumArray(size_t(s.nOutputChannel), 0.0, 0.2);
  vec interimOutput(size_t(s.nInputChannel) * s.shape.x * s.shape.y);
  vec output(size_t(s.nOutputChannel) * s.shape.x * s.shape.y);

  cpu::Params cpuP;

  cpuP.input          = input.data();
  cpuP.dwFilter       = dwFilter.data();
  cpuP.dwBias         = dwBias.data();
  cpuP.dFilter        = dFilter.data();
  cpuP.dBias          = dBias.data();
  cpuP.interimOutput  = interimOutput.data();
  cpuP.output         = output.data();

  cpuP.nInputChannel  = s.nInputChannel;
  cpuP.nOutputChannel = s.nOutputChannel;
  cpuP.shape          = s.shape;

  cpu::Convolution(cpuP);

  std::unique_ptr<TestData> ptr(new TestData(std::move(input),
                                             std::move(dwFilter),
                                             std::move(dwBias),
                                             std::move(dFilter),
                                             std::move(dBias),
                                             std::move(interimOutput),
                                             std::move(output)));
  return ptr;
}

TestData::TestData(
  vec i,
  vec dwf,
  vec dwb,
  vec df,
  vec db,
  vec io,
  vec o
)
  : input(std::move(i))
  , dwFilter(std::move(dwf))
  , dwBias(std::move(dwb))
  , dFilter(std::move(df))
  , dBias(std::move(db))
  , interimOutput(std::move(io))
  , output(std::move(o))
{}
