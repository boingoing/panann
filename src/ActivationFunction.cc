//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <cmath>

#include "ActivationFunction.h"

namespace panann {

double ActivationFunction::ExecuteSigmoidSymmetric(double value) {
    return std::tanh(value);
}

double ActivationFunction::ExecuteDerivativeSigmoidSymmetric(double value) {
    static constexpr double clamp_min = -0.98;
    static constexpr double clamp_max = 0.98;
    value = std::clamp(value, clamp_min, clamp_max);
    return 1.0 - (value * value);
}

double ActivationFunction::ExecuteSigmoid(double value) {
    return 1.0 / (1.0 + std::exp(-2 * value));
}

double ActivationFunction::ExecuteDerivativeSigmoid(double value) {
    static constexpr double clamp_min = 0.01;
    static constexpr double clamp_max = 0.99;
    value = std::clamp(value, clamp_min, clamp_max);
    return 2 * value * (1.0 - value);
}

double ActivationFunction::ExecuteLinear(double value) {
    return value;
}

double ActivationFunction::ExecuteDerivativeLinear(double /*unused*/) {
    return 1;
}

double ActivationFunction::ExecuteGaussian(double value) {
    return std::exp(-value * value);
}

double ActivationFunction::ExecuteDerivativeGaussian(double value, double field) {
    return -2 * field * value;
}

double ActivationFunction::ExecuteGaussianSymmetric(double value) {
    return (2 * std::exp(-value * value)) - 1.0;
}

double ActivationFunction::ExecuteDerivativeGaussianSymmetric(double value, double field) {
    return -2 * field * (value + 1.0);
}

double ActivationFunction::ExecuteSine(double value) {
    static constexpr double shift = 0.5;
    return std::sin(value) / 2 + shift;
}

double ActivationFunction::ExecuteDerivativeSine(double value) {
    return std::cos(value) / 2;
}

double ActivationFunction::ExecuteCosine(double value) {
    static constexpr double shift = 0.5;
    return std::cos(value) / 2 + shift;
}

double ActivationFunction::ExecuteDerivativeCosine(double value) {
    return -std::sin(value) / 2;
}

double ActivationFunction::ExecuteElliot(double value) {
    static constexpr double shift = 0.5;
    return (value / 2) / (1.0 + std::fabs(value)) + shift;
}

double ActivationFunction::ExecuteDerivativeElliot(double field) {
    static constexpr double clamp_min = 0.01;
    static constexpr double clamp_max = 0.99;
    field = std::clamp(field, clamp_min, clamp_max);
    const double abs_plus_one = std::fabs(field) + 1.0;
    return 1.0 / (2 * abs_plus_one * abs_plus_one);
}

double ActivationFunction::ExecuteElliotSymmetric(double value) {
    return value / (1.0 + std::fabs(value));
}

double ActivationFunction::ExecuteDerivativeElliotSymmetric(double field) {
    static constexpr double clamp_min = -0.98;
    static constexpr double clamp_max = 0.98;
    field = std::clamp(field, clamp_min, clamp_max);
    const double abs_plus_one = std::fabs(field) + 1.0;
    return 1.0 / (abs_plus_one * abs_plus_one);
}

double ActivationFunction::ExecuteSineSymmetric(double value) {
    return std::sin(value);
}

double ActivationFunction::ExecuteDerivativeSineSymmetric(double value) {
    return std::cos(value);
}

double ActivationFunction::ExecuteCosineSymmetric(double value) {
    return std::cos(value);
}

double ActivationFunction::ExecuteDerivativeCosineSymmetric(double value) {
    return -std::sin(value);
}

double ActivationFunction::ExecuteThreshold(double value) {
    static constexpr double threshold = 0.0;
    return value < threshold ? 0 : 1;
}

double ActivationFunction::ExecuteThresholdSymmetric(double value) {
    static constexpr double threshold = 0.0;
    return value < threshold ? -1 : 1;
}

}  // namespace panann
