#include "utils.h"

std::vector<float> GenerateRandomNumArray(
  size_t size, 
  float from, 
  float to
) 
{
  std::random_device r;
  std::default_random_engine e1(r());

  std::vector<float> v(size);

  for (auto& p : v) {
    p = std::uniform_real_distribution<float>(from, to)(e1);
  }

  return v;
}

bool Cmpf(float a, float b, float epsilon)
{
  return (fabs(a - b) < epsilon);
}

bool AreEqual(const std::vector<float>& a, const std::vector<float>& b, float epsilon) 
{
  if (a.size() != b.size()) {
    return false;
  }

  for (size_t i = 0; i < a.size(); ++i) {
    if (!Cmpf(a[i], b[i], epsilon)) {
      return false;
    }
  }
  return true;
}
