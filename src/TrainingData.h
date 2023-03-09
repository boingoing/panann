//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#ifndef TRAININGDATA_H__
#define TRAININGDATA_H__

#include <cstdint>
#include <vector>

namespace panann {

/**
 * An example used to train a neural network.<br/>
 * Feeding the input values into a network and running forward should produce the output.
 */
struct Example {
    std::vector<double> input;
    std::vector<double> output;
};

/**
 * The TrainingData is just a set of Examples with some utility functions for scaling that data.<br/>
 * @see Example
 */
class TrainingData : public std::vector<Example> {
public:
    enum class ScalingAlgorithm : uint8_t {
        /**
         * Perform no scaling of the Examples.
         */
        None = 0,

        /**
         * Scale the Examples by calculating a simple factor which will shift the values in the
         * Examples into the range [simple_scaling_new_min_, simple_scaling_new_max_] (which is [-1,1]
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

    TrainingData() = default;
    TrainingData(const TrainingData&) = delete;
    TrainingData& operator=(const TrainingData&) = delete;
    ~TrainingData() = default;

    /**
     * Set the algorithm used to scale Examples in this TrainingData via Scale().<br/>
     * Default: ScalingAlgorithm::Simple
     * @see ScalingAlgorithm
     * @see Scale
     */
    void SetScalingAlgorithm(ScalingAlgorithm algorithm);
    ScalingAlgorithm GetScalingAlgorithm() const;

    /**
     * Set the new minimum value used by the simple scaling algorithm.<br/>
     * When using simple scaling, Examples will be scaled to the range
     * [simple_scaling_new_min_, simple_scaling_new_max_].<br/>
     * By default, this range is [-1, 1].
     * @see SetSimpleScalingNewMax
     */
    void SetSimpleScalingNewMin(double val);
    double GetSimpleScalingNewMin() const;

    /**
     * Set the new maximum value used by the simple scaling algorithm.<br/>
     * When using simple scaling, Examples will be scaled to the range
     * [simple_scaling_new_min_, simple_scaling_new_max_].<br/>
     * By default, this range is [-1, 1].
     * @see SetSimpleScalingNewMin
     */
    void SetSimpleScalingNewMax(double val);
    double GetSimpleScalingNewMax() const;

    /**
     * Set the multiplier used by standard deviation scaling algorithm.<br/>
     * After recentering the Examples around their mean, we scale each
     * value by this multiple of the standard deviation.<br/>
     * Default: 2.5
     */
    void SetStandardDeviationMultiplier(double val);
    double GetStandardDeviationMultiplier() const;

    /**
     * Set the multiplier used by uniform norm scaling algorithm.<br/>
     * After recentering the Examples around their mean, we scale each
     * value by this multiple of the uniform norm.<br/>
     * Default: 1.0
     */
    void SetUniformNormMultiplier(double val);
    double GetUniformNormMultiplier() const;

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
    void ScaleInput(std::vector<double>* vec) const;

    /**
     * Scale one vector of output.<br/>
     * Uses the scaling parameters calculated during a previous call to Scale.
     * @see Scale
     */
    void ScaleOutput(std::vector<double>* vec) const;

    /**
     * Descale one vector of input.<br/>
     * Uses the scaling parameters calculated during a previous call to Scale.
     * @see Scale
     */
    void DescaleInput(std::vector<double>* vec) const;

    /**
     * Descale one vector of input.<br/>
     * Uses the scaling parameters calculated during a previous call to Scale.
     * @see Scale
     */
    void DescaleOutput(std::vector<double>* vec) const;

    /**
     * Convert sequential data into examples.<br/>
     * Use this to create examples from time series data or other sets of sequential data.<br/>
     * Each example will have input_length input samples and one output sample - which will be the
     * value from data immediately following those input samples.<br/>
     * We will create as many examples as possible from the data.
     * @param input_length The number of input samples to put into each example.
     * @param data An ordered set of samples. Must have at least input_length elements.
     */
    void FromSequentialData(const std::vector<double>& data, size_t input_length);

protected:
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

private:
static constexpr double DefaultSimpleScalingNewMin = -1.0;
static constexpr double DefaultSimpleScalingNewMax = 1.0;
    static constexpr double DefaultStandardDeviationMultiplier = 2.5;
    static constexpr double DefaultUniformNormMultiplier = 1.0;

    double simple_scaling_new_min_ = DefaultSimpleScalingNewMin;
    double simple_scaling_new_max_ = DefaultSimpleScalingNewMax;
    double input_old_min_ = 0;
    double input_old_max_ = 0;
    double output_old_min_ = 0;
    double output_old_max_ = 0;
    double input_factor_ = 0;
    double output_factor_ = 0;
    double input_mean_ = 0;
    double input_standard_deviation_ = 0;
    double output_mean_ = 0;
    double output_standard_deviation_ = 0;
    double standard_deviation_multiplier_ = DefaultStandardDeviationMultiplier;
    double input_uniform_norm_ = 0;
    double output_uniform_norm_ = 0;
    double uniform_norm_multiplier_ = DefaultUniformNormMultiplier;
    ScalingAlgorithm scaling_algorithm_ = ScalingAlgorithm::Simple;
};

} // namespace panann

#endif  // TRAININGDATA_H__
