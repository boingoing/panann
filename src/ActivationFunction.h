//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#ifndef ACTIVATIONFUNCTION_H__
#define ACTIVATIONFUNCTION_H__

namespace panann {

enum class ActivationFunctionType : uint8_t {
    Linear = 1,
    Sigmoid,
    Gaussian,
    Sine,
    Cosine,
    Elliot,
    Threshold,
    SigmoidSymmetric,
    GaussianSymmetric,
    SineSymmetric,
    CosineSymmetric,
    ElliotSymmetric,
    ThresholdSymmetric,

    // The symmetric activation functions are grouped at the end so we would
    // know that an activation function is symmetric if it's at least the
    // first one in this group.
    FirstSymmetric = SigmoidSymmetric
};

class ActivationFunction
{
public:
    static double ExecuteLinear(double value);
    static double ExecuteDerivativeLinear(double value);

    static double ExecuteSigmoid(double value);
    static double ExecuteDerivativeSigmoid(double value);

    static double ExecuteSigmoidSymmetric(double value);
    static double ExecuteDerivativeSigmoidSymmetric(double value);

    static double ExecuteGaussian(double value);
    static double ExecuteDerivativeGaussian(double value, double field);

    static double ExecuteGaussianSymmetric(double value);
    static double ExecuteDerivativeGaussianSymmetric(double value, double field);

    static double ExecuteSine(double value);
    static double ExecuteDerivativeSine(double value);

    static double ExecuteSineSymmetric(double value);
    static double ExecuteDerivativeSineSymmetric(double value);

    static double ExecuteCosine(double value);
    static double ExecuteDerivativeCosine(double value);

    static double ExecuteCosineSymmetric(double value);
    static double ExecuteDerivativeCosineSymmetric(double value);

    static double ExecuteElliot(double value);
    static double ExecuteDerivativeElliot(double field);

    static double ExecuteElliotSymmetric(double value);
    static double ExecuteDerivativeElliotSymmetric(double field);

    static double ExecuteThreshold(double value);

    static double ExecuteThresholdSymmetric(double value);
};

} // namespace panann

#endif  // ACTIVATIONFUNCTION_H__
