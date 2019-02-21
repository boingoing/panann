//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#pragma once

#include <vector>

namespace panann {

/**
 * An example used to train a neural network.<br/>
 * Feeding the input values into a network and running forward should produce the output.
 */
struct Example {
    std::vector<double> _input;
    std::vector<double> _output;
};

/**
 * The TrainingData is just a set of Examples with some utility functions for scaling that data.<br/>
 * @see Example
 */
class TrainingData : public std::vector<Example> {
public:
    enum ScalingAlgorithm : uint8_t {
        /**
         * Perform no scaling of the Examples.
         */
        None = 0,

        /**
         * Scale the Examples by calculating a simple factor which will shift the values in the
         * Examples into the range [_simpleScalingNewMin, _simpleScalingNewMax] (which is [-1,1]
         * by default).
         * @see SetSimpleScalingNewMin
         * @see SetSimpleScalingNewMax
         */
        Simple,

        /**
         * Recenters the Examples around the mean and scales them by a factor of the standard
         * deviation.
         * @see SetStandardDeviationMultiplier
         */
        StandardDeviation,

        /**
         * Recenters the Examples around the mean and scales them by a factor of the uniform
         * norm.
         * @see SetUniformNormMultiplier
         */
        UniformNorm
    };

    TrainingData();

    /**
     * Set the algorithm used to scale Examples in this TrainingData via Scale().<br/>
     * Default: ScalingAlgorithm::Simple
     * @see ScalingAlgorithm
     * @see Scale
     */
    void SetScalingAlgorithm(ScalingAlgorithm algorithm);
    ScalingAlgorithm GetScalingAlgorithm();

    /**
     * Set the new minimum value used by the simple scaling algorithm.<br/>
     * When using simple scaling, Examples will be scaled to the range
     * [_simpleScalingNewMin, _simpleScalingNewMax].<br/>
     * By default, this range is [-1, 1].
     * @see SetSimpleScalingNewMax
     */
    void SetSimpleScalingNewMin(double val);
    double GetSimpleScalingNewMin();

    /**
     * Set the new maximum value used by the simple scaling algorithm.<br/>
     * When using simple scaling, Examples will be scaled to the range
     * [_simpleScalingNewMin, _simpleScalingNewMax].<br/>
     * By default, this range is [-1, 1].
     * @see SetSimpleScalingNewMin
     */
    void SetSimpleScalingNewMax(double val);
    double GetSimpleScalingNewMax();

    /**
     * Set the multiplier used by standard deviation scaling algorithm.<br/>
     * After recentering the Examples around their mean, we scale each
     * value by this multiple of the standard deviation.<br/>
     * Default: 2.5
     */
    void SetStandardDeviationMultiplier(double val);
    double GetStandardDeviationMultiplier();

    /**
     * Set the multiplier used by uniform norm scaling algorithm.<br/>
     * After recentering the Examples around their mean, we scale each
     * value by this multiple of the uniform norm.<br/>
     * Default: 1.0
     */
    void SetUniformNormMultiplier(double val);
    double GetUniformNormMultiplier();

    /**
     * Scale the Examples in this TrainingData via the selected scaling algorithm.<br/>
     * Use SetScalingAlgorithm to select a scaling algorithm.<br/>
     * The scaling parameters are calculated separately for input and output because these
     * two sets may be in completely different scales.
     * @see SetScalingAlgorithm
     * @see Descale
     */
    void Scale();

    /**
     * Descale the Examples in this TrainingData back to their original values before the
     * data was scaled via Scale().<br/>
     * Uses the selected scaling algorithm.
     * @see SetScalingAlgorithm
     * @see Scale
     */
    void Descale();

    /**
     * Scale one vector of input.<br/>
     * Uses the scaling parameters calculated during a previous call to Scale.
     * @see Scale
     */
    void ScaleInput(std::vector<double>* vec);

    /**
     * Scale one vector of output.<br/>
     * Uses the scaling parameters calculated during a previous call to Scale.
     * @see Scale
     */
    void ScaleOutput(std::vector<double>* vec);

    /**
     * Descale one vector of input.<br/>
     * Uses the scaling parameters calculated during a previous call to Scale.
     * @see Scale
     */
    void DescaleInput(std::vector<double>* vec);

    /**
     * Descale one vector of input.<br/>
     * Uses the scaling parameters calculated during a previous call to Scale.
     * @see Scale
     */
    void DescaleOutput(std::vector<double>* vec);

protected:
    TrainingData(const TrainingData&);

    ScalingAlgorithm _scalingAlgorithm;
    double _simpleScalingNewMin;
    double _simpleScalingNewMax;
    double _inputOldMin;
    double _inputOldMax;
    double _outputOldMin;
    double _outputOldMax;
    double _inputFactor;
    double _outputFactor;
    double _inputMean;
    double _inputStandardDeviation;
    double _outputMean;
    double _outputStandardDeviation;
    double _standardDeviationMultiplier;
    double _inputUniformNorm;
    double _outputUniformNorm;
    double _uniformNormMultiplier;

    void CalculateMinMax();
    void CalculateMean();
    void CalculateStdev();
    void CalculateUniformNorm();

    void ScaleSimple();
    void ScaleStandardDeviation();
    void ScaleUniformNorm();

    void DescaleSimple();
    void DescaleStandardDeviation();
    void DescaleUniformNorm();
};

} // namespace panann
