//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>

#include "TrainingData.h"

namespace {

void SimpleScale(std::vector<double>* vec, double old_min, double factor, double new_min) {
    for (auto& val : *vec) {
        val = (val - old_min) * factor + new_min;
    }
}

void SimpleDescale(std::vector<double>* vec, double old_min, double factor, double new_min) {
    for (auto& val : *vec) {
        val = (val - new_min) / factor + old_min;
    }
}

void StdevScale(std::vector<double>* vec, double mean, double stdev, double multiplier) {
    for (auto& val : *vec) {
        val = (val - mean) / (stdev * multiplier);
    }
}

void StdevDescale(std::vector<double>* vec, double mean, double stdev, double multiplier) {
    for (auto& val : *vec) {
        val = (val * stdev * multiplier) + mean;
    }
}

void UniformNormScale(std::vector<double>* vec, double mean, double uniform_norm, double multiplier) {
    for (auto& val : *vec) {
        val = (val - mean) / (uniform_norm * multiplier);
    }
}

void UniformNormDescale(std::vector<double>* vec, double mean, double uniform_norm, double multiplier) {
    for (auto& val : *vec) {
        val = (val * uniform_norm * multiplier) + mean;
    }
}

double UniformNorm(const std::vector<double>& vec, double mean) {
    const double& val = *std::max_element(vec.cbegin(), vec.cend(), [&](const double& left, const double& right) {
        return std::fabs(left - mean) < std::fabs(right - mean);
    });
    return std::fabs(val - mean);
}

}  // namespace

namespace panann {

void TrainingData::SetScalingAlgorithm(ScalingAlgorithm algorithm) {
    this->scaling_algorithm_ = algorithm;
}

TrainingData::ScalingAlgorithm TrainingData::GetScalingAlgorithm() const {
    return this->scaling_algorithm_;
}

void TrainingData::SetSimpleScalingNewMin(double val) {
    this->simple_scaling_new_min_ = val;
}

double TrainingData::GetSimpleScalingNewMin() const {
    return this->simple_scaling_new_min_;
}

void TrainingData::SetSimpleScalingNewMax(double val) {
    this->simple_scaling_new_max_ = val;
}

double TrainingData::GetSimpleScalingNewMax() const {
    return this->simple_scaling_new_max_;
}

void TrainingData::SetStandardDeviationMultiplier(double val) {
    this->standard_deviation_multiplier_ = val;
}

double TrainingData::GetStandardDeviationMultiplier() const {
    return this->standard_deviation_multiplier_;
}

void TrainingData::SetUniformNormMultiplier(double val) {
    this->uniform_norm_multiplier_ = val;
}

double TrainingData::GetUniformNormMultiplier() const {
    return this->uniform_norm_multiplier_;
}

void TrainingData::CalculateMean() {
    double sum_input = 0.0;
    double sum_output = 0.0;
    size_t count_input = 0;
    size_t count_output = 0;

    for (const auto& example : *this) {
        sum_input = std::accumulate(example.input.cbegin(), example.input.cend(), sum_input);
        sum_output = std::accumulate(example.output.cbegin(), example.output.cend(), sum_output);
        count_input += example.input.size();
        count_output += example.output.size();
    }

    assert(count_input > 0);
    input_mean_ = sum_input / count_input;
    assert(count_output > 0);
    output_mean_ = sum_output / count_output;
}

void TrainingData::CalculateUniformNorm() {
    input_uniform_norm_ = std::numeric_limits<double>::min();
    output_uniform_norm_ = std::numeric_limits<double>::min();

    for (const auto& example : *this) {
        input_uniform_norm_ = std::max(input_uniform_norm_, UniformNorm(example.input, input_mean_));
        output_uniform_norm_ = std::max(output_uniform_norm_, UniformNorm(example.output, output_mean_));
    }
}

void TrainingData::CalculateStdev() {
    double sum_input = 0.0;
    double sum_output = 0.0;
    size_t count_input = 0;
    size_t count_output = 0;

    for (const auto& example : *this) {
        for (const auto& val : example.input) {
            sum_input += (val - input_mean_) * (val - input_mean_);
        }
        count_input += example.input.size();

        for (const auto& val : example.output) {
            sum_output += (val - output_mean_) * (val - output_mean_);
        }
        count_output += example.output.size();
    }

    assert(count_input > 1);
    input_standard_deviation_ = std::sqrt(sum_input / (count_input - 1));
    assert(count_output > 1);
    output_standard_deviation_ = std::sqrt(sum_output / (count_output - 1));
}

void TrainingData::CalculateMinMax() {
    output_old_min_ = input_old_min_ = std::numeric_limits<double>::max();
    output_old_max_ = input_old_max_ = std::numeric_limits<double>::min();

    for (const auto& example : *this) {
            auto [input_min, input_max] = std::minmax_element(example.input.cbegin(), example.input.cend());
            input_old_min_ = std::min(input_old_min_, *input_min);
            input_old_max_ = std::max(input_old_max_, *input_max);

            auto [output_min, output_max] = std::minmax_element(example.output.cbegin(), example.output.cend());
            output_old_min_ = std::min(output_old_min_, *output_min);
            output_old_max_ = std::max(output_old_max_, *output_max);
    }

    const double new_span = simple_scaling_new_max_ - simple_scaling_new_min_;
    const double input_old_span = input_old_max_ - input_old_min_;
    input_factor_ = new_span / input_old_span;

    const double output_old_span = output_old_max_ - output_old_min_;
    output_factor_ = new_span / output_old_span;
}

void TrainingData::ScaleSimple() {
    CalculateMinMax();

    for (auto& example : *this) {
        SimpleScale(&example.input, input_old_min_, input_factor_, simple_scaling_new_min_);
        SimpleScale(&example.output, output_old_min_, output_factor_, simple_scaling_new_min_);
    }
}

void TrainingData::ScaleStandardDeviation() {
    CalculateMean();
    CalculateStdev();

    for (auto& example : *this) {
        StdevScale(&example.input, input_mean_, input_standard_deviation_, standard_deviation_multiplier_);
        StdevScale(&example.output, output_mean_, output_standard_deviation_, standard_deviation_multiplier_);
    }
}

void TrainingData::ScaleUniformNorm() {
    CalculateMean();
    CalculateUniformNorm();

    for (auto& example : *this) {
        UniformNormScale(&example.input, input_mean_, input_uniform_norm_, uniform_norm_multiplier_);
        UniformNormScale(&example.output, output_mean_, output_uniform_norm_, uniform_norm_multiplier_);
    }
}

void TrainingData::DescaleSimple() {
    for (auto& example : *this) {
        SimpleDescale(&example.input, input_old_min_, input_factor_, simple_scaling_new_min_);
        SimpleDescale(&example.output, output_old_min_, output_factor_, simple_scaling_new_min_);
    }
}

void TrainingData::DescaleStandardDeviation() {
    for (auto& example : *this) {
        StdevDescale(&example.input, input_mean_, input_standard_deviation_, standard_deviation_multiplier_);
        StdevDescale(&example.output, output_mean_, output_standard_deviation_, standard_deviation_multiplier_);
    }
}

void TrainingData::DescaleUniformNorm() {
    for (auto& example : *this) {
        UniformNormDescale(&example.input, input_mean_, input_uniform_norm_, uniform_norm_multiplier_);
        UniformNormDescale(&example.output, output_mean_, output_uniform_norm_, uniform_norm_multiplier_);
    }
}

void TrainingData::Scale() {
    switch (scaling_algorithm_) {
    case ScalingAlgorithm::None:
        return;
    case ScalingAlgorithm::Simple:
        return ScaleSimple();
    case ScalingAlgorithm::StandardDeviation:
        return ScaleStandardDeviation();
    case ScalingAlgorithm::UniformNorm:
        return ScaleUniformNorm();
    default:
        assert(false);
    }
}

void TrainingData::Descale() {
    switch (scaling_algorithm_) {
    case ScalingAlgorithm::None:
        return;
    case ScalingAlgorithm::Simple:
        return DescaleSimple();
    case ScalingAlgorithm::StandardDeviation:
        return DescaleStandardDeviation();
    case ScalingAlgorithm::UniformNorm:
        return DescaleUniformNorm();
    default:
        assert(false);
    }
}

void TrainingData::ScaleInput(std::vector<double>* vec) const {
    switch (scaling_algorithm_) {
    case ScalingAlgorithm::None:
        return;
    case ScalingAlgorithm::Simple:
        return SimpleScale(vec, input_old_min_, input_factor_, simple_scaling_new_min_);
    case ScalingAlgorithm::StandardDeviation:
        return StdevScale(vec, input_mean_, input_standard_deviation_, standard_deviation_multiplier_);
    case ScalingAlgorithm::UniformNorm:
        return UniformNormScale(vec, input_mean_, input_uniform_norm_, uniform_norm_multiplier_);
    default:
        assert(false);
    }
}

void TrainingData::ScaleOutput(std::vector<double>* vec) const {
    switch (scaling_algorithm_) {
    case ScalingAlgorithm::None:
        return;
    case ScalingAlgorithm::Simple:
        return SimpleScale(vec, output_old_min_, output_factor_, simple_scaling_new_min_);
    case ScalingAlgorithm::StandardDeviation:
        return StdevScale(vec, output_mean_, output_standard_deviation_, standard_deviation_multiplier_);
    case ScalingAlgorithm::UniformNorm:
        return UniformNormScale(vec, output_mean_, output_uniform_norm_, uniform_norm_multiplier_);
    default:
        assert(false);
    }
}

void TrainingData::DescaleInput(std::vector<double>* vec) const {
    switch (scaling_algorithm_) {
    case ScalingAlgorithm::None:
        return;
    case ScalingAlgorithm::Simple:
        return SimpleDescale(vec, input_old_min_, input_factor_, simple_scaling_new_min_);
    case ScalingAlgorithm::StandardDeviation:
        return StdevDescale(vec, input_mean_, input_standard_deviation_, standard_deviation_multiplier_);
    case ScalingAlgorithm::UniformNorm:
        return UniformNormDescale(vec, input_mean_, input_uniform_norm_, uniform_norm_multiplier_);
    default:
        assert(false);
    }
}

void TrainingData::DescaleOutput(std::vector<double>* vec) const {
    switch (scaling_algorithm_) {
    case ScalingAlgorithm::None:
        return;
    case ScalingAlgorithm::Simple:
        return SimpleDescale(vec, output_old_min_, output_factor_, simple_scaling_new_min_);
    case ScalingAlgorithm::StandardDeviation:
        return StdevDescale(vec, output_mean_, output_standard_deviation_, standard_deviation_multiplier_);
    case ScalingAlgorithm::UniformNorm:
        return UniformNormDescale(vec, output_mean_, output_uniform_norm_, uniform_norm_multiplier_);
    default:
        assert(false);
    }
}

void TrainingData::FromSequentialData(const std::vector<double>& data, size_t input_length) {
    assert(data.size() > input_length);

    const size_t num_examples = data.size() - input_length;
    resize(num_examples);

    for (size_t i = 0; i < num_examples; i++) {
        const auto begin = data.cbegin() + i;
        const auto end = begin + (input_length - 1);

        Example& example = at(i);
        example.input.assign(begin, end);
        example.output = { data[i + input_length - 1] };
    }
}

}  // namespace panann
