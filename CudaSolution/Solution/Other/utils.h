#pragma once

#include <vector>
#include <random>

std::vector<float> GenerateRandomNumArray(size_t size,
                                          float from = -1.0,
                                          float to = +1.0);

bool AreEqual(const std::vector<float>& a, 
              const std::vector<float>& b, 
              float epsilon = 0.005f);
