#include <algorithm>

#include "ActivationFunction.h"

double ActivationFunction::ExecuteSigmoidSymmetric(double sum, double, double slope) {
    return tanh(slope * sum);
}

double ActivationFunction::ExecuteDerivativeSigmoidSymmetric(double value, double, double slope) {
    value = std::clamp(value, -0.98, 0.98);
    return slope * (1.0 - (value * value));
}
