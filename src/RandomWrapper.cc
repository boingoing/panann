//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panga contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for
// full license information.
//-------------------------------------------------------------------------------------------------------

#include "RandomWrapper.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>

namespace panann {

bool RandomWrapper::CoinFlip(double probability) {
  std::bernoulli_distribution dist(probability);
  return dist(Engine());
}

std::byte RandomWrapper::RandomByte() {
  constexpr auto max_byte = 0xff;
  auto& engine = Engine();
  const auto val = engine();
  return static_cast<std::byte>(val & max_byte);
}

std::mt19937& RandomWrapper::Engine() {
  if (!engine_.has_value()) {
    constexpr size_t seed_size = 624U;
    std::array<int, seed_size> seed_data;
    std::random_device r;
    std::generate_n(seed_data.data(), seed_data.size(), std::ref(r));
    std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
    const std::mt19937 eng(seq);
    engine_ = eng;
  }
  return *engine_;
}

}  // namespace panga
