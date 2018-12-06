#pragma once

class ActivationFunction
{
public:
    static double ExecuteSigmoidSymmetric(double value);
    static double ExecuteDerivativeSigmoidSymmetric(double value);

    static double ExecuteSigmoid(double value);
    static double ExecuteDerivativeSigmoid(double value);
};
