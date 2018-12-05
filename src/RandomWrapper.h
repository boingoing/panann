//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panga contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#pragma once

#include <random>
#include <cstddef>
#include <cstdint>
#include <array>
#include <functional>
#include <vector>
#include <algorithm>

//namespace panga {

class RandomWrapper {
private:
    std::mt19937 _engine;

public:
    RandomWrapper() {
        std::array<int, 624> seed_data;
        std::random_device r;
        std::generate_n(seed_data.data(), seed_data.size(), std::ref(r));
        std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
        std::mt19937 eng(seq);
        this->_engine = eng;
    }

    template <typename IntegerType = uint32_t>
    IntegerType RandomInteger(IntegerType min, IntegerType max) {
        std::uniform_int_distribution<IntegerType> dist(min, max);
        return dist(this->_engine);
    }

    template <typename FloatType = double>
    FloatType RandomFloat(FloatType min, FloatType max) {
        std::uniform_real_distribution<FloatType> dist(min, max);
        return dist(this->_engine);
    }

    bool CoinFlip(double probability) {
        std::bernoulli_distribution dist{ probability };
        return dist(this->_engine);
    }

    std::byte RandomByte() {
        return static_cast<std::byte>(this->_engine());
    }

    template <typename T>
    void ShuffleVector(std::vector<T>* vec) {
        std::shuffle(vec->begin(), vec->end(), this->_engine);
    }
};

//} // namespace panga
