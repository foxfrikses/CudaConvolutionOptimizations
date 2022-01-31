#pragma once

#include <vector>
#include <memory>

#include <vector_types.h>

struct TestSize {
  int nInputChannel = 3;
  int nOutputChannel = 16;
  int2 shape = {256, 256};
};

struct TestData final {
  using vec = std::vector<float>;

  const vec input;
  const vec dwFilter;
  const vec dwBias;
  const vec dFilter;
  const vec dBias;
  const vec interimOutput;
  const vec output;

  static std::unique_ptr<TestData> Generate(const TestSize&);

private:
  TestData(vec, vec, vec, vec, vec, vec, vec);
};

