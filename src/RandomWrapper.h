//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panga contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for
// full license information.
//-------------------------------------------------------------------------------------------------------

#ifndef RANDOMWRAPPER_H__
#define RANDOMWRAPPER_H__

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

namespace panann {

/**
 * A simple wrapper which may be used to generate random values.
 */
class RandomWrapper {
 public:
  RandomWrapper() = default;
  RandomWrapper(const RandomWrapper& rhs) = delete;
  RandomWrapper& operator=(const RandomWrapper& rhs) = delete;
  ~RandomWrapper() = default;

  /**
   * Generates a uniformly random integer of type |IntegerType| in the range
   * [|min|, |max|].
   */
  template <typename IntegerType = uint32_t>
  IntegerType RandomInteger(IntegerType min, IntegerType max) {
    std::uniform_int_distribution<IntegerType> dist(min, max);
    return dist(Engine());
  }

  /**
   * Generates a uniformly random floating point value of type |FloatType| in
   * the range [|min|, |max|).
   */
  template <typename FloatType = double>
  FloatType RandomFloat(FloatType min, FloatType max) {
    std::uniform_real_distribution<FloatType> dist(min, max);
    return dist(Engine());
  }

  /**
   * Generates a weighted coin flip with |probability| chance of producing the
   * value true.
   */
  bool CoinFlip(double probability);

  /**
   * Generates a uniformly random byte.
   */
  std::byte RandomByte();

    /**
     * Shuffle the elements of |vec| into random positions.
     */
    template <typename T>
    void ShuffleVector(std::vector<T>* vec) {
        std::shuffle(vec->begin(), vec->end(), Engine());
    }

 protected:
  std::mt19937& Engine();

 private:
  std::optional<std::mt19937> engine_;
};

}  // namespace panann

#endif  // RANDOMWRAPPER_H__
