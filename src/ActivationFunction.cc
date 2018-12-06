#include <algorithm>
#include <cmath>

#include "ActivationFunction.h"

double ActivationFunction::ExecuteSigmoidSymmetric(double value) {
    return std::tanh(value);
}

double ActivationFunction::ExecuteDerivativeSigmoidSymmetric(double value) {
    value = std::clamp(value, -0.98, 0.98);
    return 1.0 - (value * value);
}

double ActivationFunction::ExecuteSigmoid(double value) {
    return 1.0 / (1.0 + std::exp(-2.0 * value));
}

double ActivationFunction::ExecuteDerivativeSigmoid(double value) {
    value = std::clamp(value, 0.01, 0.99);
    return 2.0 * value * (1.0 - value);
}
