#pragma once

class ActivationFunction
{
public:
    static double ExecuteSigmoidSymmetric(double sum, double threshold = 0.0, double slope = 1.0);
    static double ExecuteDerivativeSigmoidSymmetric(double value, double sum, double slope = 1.0);
};
