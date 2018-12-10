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

double ActivationFunction::ExecuteLinear(double value) {
    return value;
}

double ActivationFunction::ExecuteDerivativeLinear(double) {
    return 1;
}

double ActivationFunction::ExecuteGaussian(double value) {
    return std::exp(-value * value);
}

double ActivationFunction::ExecuteDerivativeGaussian(double value, double field) {
    return -2.0 * field * value;
}

double ActivationFunction::ExecuteGaussianSymmetric(double value) {
    return (2.0 * std::exp(-value * value)) - 1.0f;
}

double ActivationFunction::ExecuteDerivativeGaussianSymmetric(double value, double field) {
    return -2.0 * field * (value + 1.0);
}

double ActivationFunction::ExecuteSine(double value) {
    return std::sin(value) / 2.0 + 0.5;
}

double ActivationFunction::ExecuteDerivativeSine(double value) {
    return std::cos(value) / 2.0;
}

double ActivationFunction::ExecuteCosine(double value) {
    return std::cos(value) / 2.0 + 0.5;
}

double ActivationFunction::ExecuteDerivativeCosine(double value) {
    return -std::sin(value) / 2.0;
}

double ActivationFunction::ExecuteElliot(double value) {
    return (value / 2.0) / (1.0 + std::fabs(value)) + 0.5;
}

double ActivationFunction::ExecuteDerivativeElliot(double field) {
    field = std::clamp(field, 0.01, 0.99);
    double absPlusOne = std::fabs(field) + 1.0;
    return 1.0 / (2.0 * absPlusOne * absPlusOne);
}

double ActivationFunction::ExecuteElliotSymmetric(double value) {
    return value / (1.0 + std::fabs(value));
}

double ActivationFunction::ExecuteDerivativeElliotSymmetric(double field) {
    field = std::clamp(field, -0.98, 0.98);
    double absPlusOne = std::fabs(field) + 1.0;
    return 1.0 / (absPlusOne * absPlusOne);
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
    double threshold = 0.0;
    return value < threshold ? 0 : 1;
}

double ActivationFunction::ExecuteThresholdSymmetric(double value) {
    double threshold = 0.0;
    return value < threshold ? -1 : 1;
}
