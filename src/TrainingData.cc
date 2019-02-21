//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <cassert>

#include "TrainingData.h"

using namespace panann;

TrainingData::TrainingData() : std::vector<Example>(),
    _scalingAlgorithm(ScalingAlgorithm::Simple),
    _simpleScalingNewMin(-1.0),
    _simpleScalingNewMax(1.0),
    _inputOldMin(0.0),
    _inputOldMax(0.0),
    _outputOldMin(0.0),
    _outputOldMax(0.0),
    _inputFactor(0.0),
    _outputFactor(0.0),
    _inputMean(0.0),
    _inputStandardDeviation(0.0),
    _outputMean(0.0),
    _outputStandardDeviation(0.0),
    _standardDeviationMultiplier(2.5),
    _inputUniformNorm(0.0),
    _outputUniformNorm(0.0),
    _uniformNormMultiplier(1.0) {
}

void TrainingData::SetScalingAlgorithm(ScalingAlgorithm algorithm) {
    this->_scalingAlgorithm = algorithm;
}

TrainingData::ScalingAlgorithm TrainingData::GetScalingAlgorithm() {
    return this->_scalingAlgorithm;
}

void TrainingData::SetSimpleScalingNewMin(double val) {
    this->_simpleScalingNewMin = val;
}

double TrainingData::GetSimpleScalingNewMin() {
    return this->_simpleScalingNewMin;
}

void TrainingData::SetSimpleScalingNewMax(double val) {
    this->_simpleScalingNewMax = val;
}

double TrainingData::GetSimpleScalingNewMax() {
    return this->_simpleScalingNewMax;
}

void TrainingData::SetStandardDeviationMultiplier(double val) {
    this->_standardDeviationMultiplier = val;
}

double TrainingData::GetStandardDeviationMultiplier() {
    return this->_standardDeviationMultiplier;
}

void TrainingData::SetUniformNormMultiplier(double val) {
    this->_uniformNormMultiplier = val;
}

double TrainingData::GetUniformNormMultiplier() {
    return this->_uniformNormMultiplier;
}

void SimpleScale(std::vector<double>* vec, double oldMin, double factor, double newMin) {
    for (auto it = vec->begin(); it != vec->end(); it++) {
        *it = (*it - oldMin) * factor + newMin;
    }
}

void SimpleDescale(std::vector<double>* vec, double oldMin, double factor, double newMin) {
    for (auto it = vec->begin(); it != vec->end(); it++) {
        *it = (*it - newMin) / factor + oldMin;
    }
}

void StdevScale(std::vector<double>* vec, double mean, double stdev, double multiplier) {
    for (auto it = vec->begin(); it != vec->end(); it++) {
        *it = (*it - mean) / (stdev * multiplier);
    }
}

void StdevDescale(std::vector<double>* vec, double mean, double stdev, double multiplier) {
    for (auto it = vec->begin(); it != vec->end(); it++) {
        *it = (*it * stdev * multiplier) + mean;
    }
}

void UniformNormScale(std::vector<double>* vec, double mean, double uniformNorm, double multiplier) {
    for (auto it = vec->begin(); it != vec->end(); it++) {
        *it = (*it - mean) / (uniformNorm * multiplier);
    }
}

void UniformNormDescale(std::vector<double>* vec, double mean, double uniformNorm, double multiplier) {
    for (auto it = vec->begin(); it != vec->end(); it++) {
        *it = (*it * uniformNorm * multiplier) + mean;
    }
}

void TrainingData::CalculateMean() {
    double sumInput = 0.0;
    double sumOutput = 0.0;
    size_t countInput = 0;
    size_t countOutput = 0;
    std::for_each(this->begin(), this->end(), [&](const Example& example) {
        const std::vector<double>& input = example._input;
        sumInput = std::accumulate(input.begin(), input.end(), sumInput);
        countInput += input.size();

        const std::vector<double>& output = example._input;
        sumOutput = std::accumulate(output.begin(), output.end(), sumOutput);
        countOutput += output.size();
    });

    assert(countInput > 0);
    this->_inputMean = sumInput / countInput;
    assert(countOutput > 0);
    this->_outputMean = sumOutput / countOutput;
}

double UniformNorm(const std::vector<double>* vec, double mean) {
    const double& val = *std::max_element(vec->begin(), vec->end(), [&](const double& left, const double& right) {
        return std::fabs(left - mean) < std::fabs(right - mean);
    });
    return std::fabs(val - mean);
}

void TrainingData::CalculateUniformNorm() {
    this->_inputUniformNorm = std::numeric_limits<double>::min();
    this->_outputUniformNorm = std::numeric_limits<double>::min();

    std::for_each(this->begin(), this->end(), [&](const Example& example) {
        this->_inputUniformNorm = std::max(this->_inputUniformNorm, ::UniformNorm(&example._input, this->_inputMean));
        this->_outputUniformNorm = std::max(this->_outputUniformNorm, ::UniformNorm(&example._output, this->_outputMean));
    });
}

void TrainingData::CalculateStdev() {
    double sumInput = 0.0;
    double sumOutput = 0.0;
    size_t countInput = 0;
    size_t countOutput = 0;
    std::for_each(this->begin(), this->end(), [&](const Example& example) {
        const std::vector<double>& input = example._input;
        std::for_each(input.begin(), input.end(), [&](const double& val) {
            sumInput += (val - this->_inputMean) * (val - this->_inputMean);
        });
        countInput += input.size();

        const std::vector<double>& output = example._input;
        std::for_each(output.begin(), output.end(), [&](const double& val) {
            sumOutput += (val - this->_outputMean) * (val - this->_outputMean);
        });
        countOutput += output.size();
    });

    assert(countInput > 1);
    this->_inputStandardDeviation = std::sqrt(sumInput / (countInput - 1));
    assert(countOutput > 1);
    this->_outputStandardDeviation = std::sqrt(sumOutput / (countOutput - 1));
}

void TrainingData::CalculateMinMax() {
    this->_outputOldMin = this->_inputOldMin = std::numeric_limits<double>::max();
    this->_outputOldMax = this->_inputOldMax = std::numeric_limits<double>::min();

    std::for_each(this->begin(), this->end(), [&](const Example& example) {
        {
            const std::vector<double>& input = example._input;
            auto [min, max] = std::minmax_element(input.begin(), input.end());
            this->_inputOldMin = std::min(this->_inputOldMin, *min);
            this->_inputOldMax = std::max(this->_inputOldMax, *max);
        }
        {
            const std::vector<double>& output = example._output;
            auto [min, max] = std::minmax_element(output.begin(), output.end());
            this->_outputOldMin = std::min(this->_outputOldMin, *min);
            this->_outputOldMax = std::max(this->_outputOldMax, *max);
        }
    });

    double newSpan = this->_simpleScalingNewMax - this->_simpleScalingNewMin;
    double oldSpan = this->_inputOldMax - this->_inputOldMin;
    this->_inputFactor = newSpan / oldSpan;

    oldSpan = this->_outputOldMax - this->_outputOldMin;
    this->_outputFactor = newSpan / oldSpan;
}

void TrainingData::ScaleSimple() {
    this->CalculateMinMax();

    std::for_each(this->begin(), this->end(), [&](Example& example) {
        SimpleScale(&example._input, this->_inputOldMin, this->_inputFactor, this->_simpleScalingNewMin);
        SimpleScale(&example._output, this->_outputOldMin, this->_outputFactor, this->_simpleScalingNewMin);
    });
}

void TrainingData::ScaleStandardDeviation() {
    this->CalculateMean();
    this->CalculateStdev();

    std::for_each(this->begin(), this->end(), [&](Example& example) {
        StdevScale(&example._input, this->_inputMean, this->_inputStandardDeviation, this->_standardDeviationMultiplier);
        StdevScale(&example._output, this->_outputMean, this->_outputStandardDeviation, this->_standardDeviationMultiplier);
    });
}

void TrainingData::ScaleUniformNorm() {
    this->CalculateMean();
    this->CalculateUniformNorm();

    std::for_each(this->begin(), this->end(), [&](Example& example) {
        UniformNormScale(&example._input, this->_inputMean, this->_inputUniformNorm, this->_uniformNormMultiplier);
        UniformNormScale(&example._output, this->_outputMean, this->_outputUniformNorm, this->_uniformNormMultiplier);
    });
}

void TrainingData::DescaleSimple() {
    std::for_each(this->begin(), this->end(), [&](Example& example) {
        SimpleDescale(&example._input, this->_inputOldMin, this->_inputFactor, this->_simpleScalingNewMin);
        SimpleDescale(&example._output, this->_outputOldMin, this->_outputFactor, this->_simpleScalingNewMin);
    });
}

void TrainingData::DescaleStandardDeviation() {
    std::for_each(this->begin(), this->end(), [&](Example& example) {
        StdevDescale(&example._input, this->_inputMean, this->_inputStandardDeviation, this->_standardDeviationMultiplier);
        StdevDescale(&example._output, this->_outputMean, this->_outputStandardDeviation, this->_standardDeviationMultiplier);
    });
}

void TrainingData::DescaleUniformNorm() {
    std::for_each(this->begin(), this->end(), [&](Example& example) {
        UniformNormDescale(&example._input, this->_inputMean, this->_inputUniformNorm, this->_uniformNormMultiplier);
        UniformNormDescale(&example._output, this->_outputMean, this->_outputUniformNorm, this->_uniformNormMultiplier);
    });
}

void TrainingData::Scale() {
    switch (this->_scalingAlgorithm) {
    case ScalingAlgorithm::None:
        return;
    case ScalingAlgorithm::Simple:
        return this->ScaleSimple();
    case ScalingAlgorithm::StandardDeviation:
        return this->ScaleStandardDeviation();
    case ScalingAlgorithm::UniformNorm:
        return this->ScaleUniformNorm();
    default:
        assert(false);
    }
}

void TrainingData::Descale() {
    switch (this->_scalingAlgorithm) {
    case ScalingAlgorithm::None:
        return;
    case ScalingAlgorithm::Simple:
        return this->DescaleSimple();
    case ScalingAlgorithm::StandardDeviation:
        return this->DescaleStandardDeviation();
    case ScalingAlgorithm::UniformNorm:
        return this->DescaleUniformNorm();
    default:
        assert(false);
    }
}

void TrainingData::ScaleInput(std::vector<double>* vec) {
    switch (this->_scalingAlgorithm) {
    case ScalingAlgorithm::None:
        return;
    case ScalingAlgorithm::Simple:
        return SimpleScale(vec, this->_inputOldMin, this->_inputFactor, this->_simpleScalingNewMin);
    case ScalingAlgorithm::StandardDeviation:
        return StdevScale(vec, this->_inputMean, this->_inputStandardDeviation, this->_standardDeviationMultiplier);
    case ScalingAlgorithm::UniformNorm:
        return UniformNormScale(vec, this->_inputMean, this->_inputUniformNorm, this->_uniformNormMultiplier);
    default:
        assert(false);
    }
}

void TrainingData::ScaleOutput(std::vector<double>* vec) {
    switch (this->_scalingAlgorithm) {
    case ScalingAlgorithm::None:
        return;
    case ScalingAlgorithm::Simple:
        return SimpleScale(vec, this->_outputOldMin, this->_outputFactor, this->_simpleScalingNewMin);
    case ScalingAlgorithm::StandardDeviation:
        return StdevScale(vec, this->_outputMean, this->_outputStandardDeviation, this->_standardDeviationMultiplier);
    case ScalingAlgorithm::UniformNorm:
        return UniformNormScale(vec, this->_outputMean, this->_outputUniformNorm, this->_uniformNormMultiplier);
    default:
        assert(false);
    }
}

void TrainingData::DescaleInput(std::vector<double>* vec) {
    switch (this->_scalingAlgorithm) {
    case ScalingAlgorithm::None:
        return;
    case ScalingAlgorithm::Simple:
        return SimpleDescale(vec, this->_inputOldMin, this->_inputFactor, this->_simpleScalingNewMin);
    case ScalingAlgorithm::StandardDeviation:
        return StdevDescale(vec, this->_inputMean, this->_inputStandardDeviation, this->_standardDeviationMultiplier);
    case ScalingAlgorithm::UniformNorm:
        return UniformNormDescale(vec, this->_inputMean, this->_inputUniformNorm, this->_uniformNormMultiplier);
    default:
        assert(false);
    }
}

void TrainingData::DescaleOutput(std::vector<double>* vec) {
    switch (this->_scalingAlgorithm) {
    case ScalingAlgorithm::None:
        return;
    case ScalingAlgorithm::Simple:
        return SimpleDescale(vec, this->_outputOldMin, this->_outputFactor, this->_simpleScalingNewMin);
    case ScalingAlgorithm::StandardDeviation:
        return StdevDescale(vec, this->_outputMean, this->_outputStandardDeviation, this->_standardDeviationMultiplier);
    case ScalingAlgorithm::UniformNorm:
        return UniformNormDescale(vec, this->_outputMean, this->_outputUniformNorm, this->_uniformNormMultiplier);
    default:
        assert(false);
    }
}
