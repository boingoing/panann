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

void TrainingData::SetScalingAlgorithm(ScalingAlgorithm algorithm) {
    this->scaling_algorithm_ = algorithm;
}

TrainingData::ScalingAlgorithm TrainingData::GetScalingAlgorithm() {
    return this->scaling_algorithm_;
}

void TrainingData::SetSimpleScalingNewMin(double val) {
    this->simple_scaling_new_min_ = val;
}

double TrainingData::GetSimpleScalingNewMin() {
    return this->simple_scaling_new_min_;
}

void TrainingData::SetSimpleScalingNewMax(double val) {
    this->simple_scaling_new_max_ = val;
}

double TrainingData::GetSimpleScalingNewMax() {
    return this->simple_scaling_new_max_;
}

void TrainingData::SetStandardDeviationMultiplier(double val) {
    this->standard_deviation_multiplier_ = val;
}

double TrainingData::GetStandardDeviationMultiplier() {
    return this->standard_deviation_multiplier_;
}

void TrainingData::SetUniformNormMultiplier(double val) {
    this->uniform_norm_multiplier_ = val;
}

double TrainingData::GetUniformNormMultiplier() {
    return this->uniform_norm_multiplier_;
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
        const std::vector<double>& input = example.input;
        sumInput = std::accumulate(input.begin(), input.end(), sumInput);
        countInput += input.size();

        const std::vector<double>& output = example.input;
        sumOutput = std::accumulate(output.begin(), output.end(), sumOutput);
        countOutput += output.size();
    });

    assert(countInput > 0);
    this->input_mean_ = sumInput / countInput;
    assert(countOutput > 0);
    this->output_mean_ = sumOutput / countOutput;
}

double UniformNorm(const std::vector<double>* vec, double mean) {
    const double& val = *std::max_element(vec->begin(), vec->end(), [&](const double& left, const double& right) {
        return std::fabs(left - mean) < std::fabs(right - mean);
    });
    return std::fabs(val - mean);
}

void TrainingData::CalculateUniformNorm() {
    this->input_uniform_norm_ = std::numeric_limits<double>::min();
    this->output_uniform_norm_ = std::numeric_limits<double>::min();

    std::for_each(this->begin(), this->end(), [&](const Example& example) {
        this->input_uniform_norm_ = std::max(this->input_uniform_norm_, ::UniformNorm(&example.input, this->input_mean_));
        this->output_uniform_norm_ = std::max(this->output_uniform_norm_, ::UniformNorm(&example.output, this->output_mean_));
    });
}

void TrainingData::CalculateStdev() {
    double sumInput = 0.0;
    double sumOutput = 0.0;
    size_t countInput = 0;
    size_t countOutput = 0;
    std::for_each(this->begin(), this->end(), [&](const Example& example) {
        const std::vector<double>& input = example.input;
        std::for_each(input.begin(), input.end(), [&](const double& val) {
            sumInput += (val - this->input_mean_) * (val - this->input_mean_);
        });
        countInput += input.size();

        const std::vector<double>& output = example.input;
        std::for_each(output.begin(), output.end(), [&](const double& val) {
            sumOutput += (val - this->output_mean_) * (val - this->output_mean_);
        });
        countOutput += output.size();
    });

    assert(countInput > 1);
    this->input_standard_deviation_ = std::sqrt(sumInput / (countInput - 1));
    assert(countOutput > 1);
    this->output_standard_deviation_ = std::sqrt(sumOutput / (countOutput - 1));
}

void TrainingData::CalculateMinMax() {
    this->output_old_min_ = this->input_old_min_ = std::numeric_limits<double>::max();
    this->output_old_max_ = this->input_old_max_ = std::numeric_limits<double>::min();

    std::for_each(this->begin(), this->end(), [&](const Example& example) {
        {
            const std::vector<double>& input = example.input;
            auto [min, max] = std::minmax_element(input.begin(), input.end());
            this->input_old_min_ = std::min(this->input_old_min_, *min);
            this->input_old_max_ = std::max(this->input_old_max_, *max);
        }
        {
            const std::vector<double>& output = example.output;
            auto [min, max] = std::minmax_element(output.begin(), output.end());
            this->output_old_min_ = std::min(this->output_old_min_, *min);
            this->output_old_max_ = std::max(this->output_old_max_, *max);
        }
    });

    double newSpan = this->simple_scaling_new_max_ - this->simple_scaling_new_min_;
    double oldSpan = this->input_old_max_ - this->input_old_min_;
    this->input_factor_ = newSpan / oldSpan;

    oldSpan = this->output_old_max_ - this->output_old_min_;
    this->output_factor_ = newSpan / oldSpan;
}

void TrainingData::ScaleSimple() {
    this->CalculateMinMax();

    std::for_each(this->begin(), this->end(), [&](Example& example) {
        SimpleScale(&example.input, this->input_old_min_, this->input_factor_, this->simple_scaling_new_min_);
        SimpleScale(&example.output, this->output_old_min_, this->output_factor_, this->simple_scaling_new_min_);
    });
}

void TrainingData::ScaleStandardDeviation() {
    this->CalculateMean();
    this->CalculateStdev();

    std::for_each(this->begin(), this->end(), [&](Example& example) {
        StdevScale(&example.input, this->input_mean_, this->input_standard_deviation_, this->standard_deviation_multiplier_);
        StdevScale(&example.output, this->output_mean_, this->output_standard_deviation_, this->standard_deviation_multiplier_);
    });
}

void TrainingData::ScaleUniformNorm() {
    this->CalculateMean();
    this->CalculateUniformNorm();

    std::for_each(this->begin(), this->end(), [&](Example& example) {
        UniformNormScale(&example.input, this->input_mean_, this->input_uniform_norm_, this->uniform_norm_multiplier_);
        UniformNormScale(&example.output, this->output_mean_, this->output_uniform_norm_, this->uniform_norm_multiplier_);
    });
}

void TrainingData::DescaleSimple() {
    std::for_each(this->begin(), this->end(), [&](Example& example) {
        SimpleDescale(&example.input, this->input_old_min_, this->input_factor_, this->simple_scaling_new_min_);
        SimpleDescale(&example.output, this->output_old_min_, this->output_factor_, this->simple_scaling_new_min_);
    });
}

void TrainingData::DescaleStandardDeviation() {
    std::for_each(this->begin(), this->end(), [&](Example& example) {
        StdevDescale(&example.input, this->input_mean_, this->input_standard_deviation_, this->standard_deviation_multiplier_);
        StdevDescale(&example.output, this->output_mean_, this->output_standard_deviation_, this->standard_deviation_multiplier_);
    });
}

void TrainingData::DescaleUniformNorm() {
    std::for_each(this->begin(), this->end(), [&](Example& example) {
        UniformNormDescale(&example.input, this->input_mean_, this->input_uniform_norm_, this->uniform_norm_multiplier_);
        UniformNormDescale(&example.output, this->output_mean_, this->output_uniform_norm_, this->uniform_norm_multiplier_);
    });
}

void TrainingData::Scale() {
    switch (this->scaling_algorithm_) {
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
    switch (this->scaling_algorithm_) {
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
    switch (this->scaling_algorithm_) {
    case ScalingAlgorithm::None:
        return;
    case ScalingAlgorithm::Simple:
        return SimpleScale(vec, this->input_old_min_, this->input_factor_, this->simple_scaling_new_min_);
    case ScalingAlgorithm::StandardDeviation:
        return StdevScale(vec, this->input_mean_, this->input_standard_deviation_, this->standard_deviation_multiplier_);
    case ScalingAlgorithm::UniformNorm:
        return UniformNormScale(vec, this->input_mean_, this->input_uniform_norm_, this->uniform_norm_multiplier_);
    default:
        assert(false);
    }
}

void TrainingData::ScaleOutput(std::vector<double>* vec) {
    switch (this->scaling_algorithm_) {
    case ScalingAlgorithm::None:
        return;
    case ScalingAlgorithm::Simple:
        return SimpleScale(vec, this->output_old_min_, this->output_factor_, this->simple_scaling_new_min_);
    case ScalingAlgorithm::StandardDeviation:
        return StdevScale(vec, this->output_mean_, this->output_standard_deviation_, this->standard_deviation_multiplier_);
    case ScalingAlgorithm::UniformNorm:
        return UniformNormScale(vec, this->output_mean_, this->output_uniform_norm_, this->uniform_norm_multiplier_);
    default:
        assert(false);
    }
}

void TrainingData::DescaleInput(std::vector<double>* vec) {
    switch (this->scaling_algorithm_) {
    case ScalingAlgorithm::None:
        return;
    case ScalingAlgorithm::Simple:
        return SimpleDescale(vec, this->input_old_min_, this->input_factor_, this->simple_scaling_new_min_);
    case ScalingAlgorithm::StandardDeviation:
        return StdevDescale(vec, this->input_mean_, this->input_standard_deviation_, this->standard_deviation_multiplier_);
    case ScalingAlgorithm::UniformNorm:
        return UniformNormDescale(vec, this->input_mean_, this->input_uniform_norm_, this->uniform_norm_multiplier_);
    default:
        assert(false);
    }
}

void TrainingData::DescaleOutput(std::vector<double>* vec) {
    switch (this->scaling_algorithm_) {
    case ScalingAlgorithm::None:
        return;
    case ScalingAlgorithm::Simple:
        return SimpleDescale(vec, this->output_old_min_, this->output_factor_, this->simple_scaling_new_min_);
    case ScalingAlgorithm::StandardDeviation:
        return StdevDescale(vec, this->output_mean_, this->output_standard_deviation_, this->standard_deviation_multiplier_);
    case ScalingAlgorithm::UniformNorm:
        return UniformNormDescale(vec, this->output_mean_, this->output_uniform_norm_, this->uniform_norm_multiplier_);
    default:
        assert(false);
    }
}

void TrainingData::FromSequentialData(std::vector<double>* data, size_t inputLength) {
    assert(data);
    assert(data->size() > inputLength);

    size_t numExamples = data->size() - inputLength;

    this->resize(numExamples);

    for (size_t i = 0; i < numExamples; i++) {
        auto begin = data->cbegin() + i;
        auto end = begin + (inputLength - 1);

        Example& example = this->operator[](i);
        example.input.assign(begin, end);
        example.output = { data->operator[](i + inputLength - 1) };
    }
}
