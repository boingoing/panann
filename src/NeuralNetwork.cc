//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <cassert>

#include "ActivationFunction.h"
#include "NeuralNetwork.h"
#include "TrainingData.h"

namespace {

using ActivationFunctionType = panann::ActivationFunctionType;
using panann::ActivationFunction;

bool IsActivationFunctionSymmetric(ActivationFunctionType type) {
    return type >= ActivationFunctionType::FirstSymmetric;
}

double ExecuteActivationFunction(ActivationFunctionType type, double value) {
    switch (type) {
    case ActivationFunctionType::Sigmoid:
        return ActivationFunction::ExecuteSigmoid(value);
    case ActivationFunctionType::SigmoidSymmetric:
        return ActivationFunction::ExecuteSigmoidSymmetric(value);
    case ActivationFunctionType::Gaussian:
        return ActivationFunction::ExecuteGaussian(value);
    case ActivationFunctionType::GaussianSymmetric:
        return ActivationFunction::ExecuteGaussianSymmetric(value);
    case ActivationFunctionType::Cosine:
        return ActivationFunction::ExecuteCosine(value);
    case ActivationFunctionType::CosineSymmetric:
        return ActivationFunction::ExecuteCosineSymmetric(value);
    case ActivationFunctionType::Sine:
        return ActivationFunction::ExecuteSine(value);
    case ActivationFunctionType::SineSymmetric:
        return ActivationFunction::ExecuteSineSymmetric(value);
    case ActivationFunctionType::Elliot:
        return ActivationFunction::ExecuteElliot(value);
    case ActivationFunctionType::ElliotSymmetric:
        return ActivationFunction::ExecuteElliotSymmetric(value);
    case ActivationFunctionType::Linear:
        return ActivationFunction::ExecuteLinear(value);
    case ActivationFunctionType::Threshold:
        return ActivationFunction::ExecuteThreshold(value);
    case ActivationFunctionType::ThresholdSymmetric:
        return ActivationFunction::ExecuteThresholdSymmetric(value);
    default:
        assert(false);
    }

    return 0;
}

double ExecuteActivationFunctionDerivative(ActivationFunctionType type, double value, double field) {
    switch (type) {
    case ActivationFunctionType::Sigmoid:
        return ActivationFunction::ExecuteDerivativeSigmoid(value);
    case ActivationFunctionType::SigmoidSymmetric:
        return ActivationFunction::ExecuteDerivativeSigmoidSymmetric(value);
    case ActivationFunctionType::Gaussian:
        return ActivationFunction::ExecuteDerivativeGaussian(value, field);
    case ActivationFunctionType::GaussianSymmetric:
        return ActivationFunction::ExecuteDerivativeGaussianSymmetric(value, field);
    case ActivationFunctionType::Cosine:
        return ActivationFunction::ExecuteDerivativeCosine(field);
    case ActivationFunctionType::CosineSymmetric:
        return ActivationFunction::ExecuteDerivativeCosineSymmetric(field);
    case ActivationFunctionType::Sine:
        return ActivationFunction::ExecuteDerivativeSine(field);
    case ActivationFunctionType::SineSymmetric:
        return ActivationFunction::ExecuteDerivativeSineSymmetric(field);
    case ActivationFunctionType::Elliot:
        return ActivationFunction::ExecuteDerivativeElliot(field);
    case ActivationFunctionType::ElliotSymmetric:
        return ActivationFunction::ExecuteDerivativeElliotSymmetric(field);
    case ActivationFunctionType::Linear:
        return ActivationFunction::ExecuteDerivativeLinear(value);
    default:
        assert(false);
    }

    return 0;
}

double ApplyErrorShaping(double value) {
    // TODO(boingoing): Should this be replaced by?
    //return tanh(value);

    static constexpr double tanh_value_limit = 0.9999999;
    static constexpr double tanh_output_clamp = 17;
    if (value < -tanh_value_limit) {
        return -tanh_output_clamp;
    }
    if (value > tanh_value_limit) {
        return tanh_output_clamp;
    }
        return log((1.0 + value) / (1.0 - value));
}

}  // namespace

namespace panann {

void NeuralNetwork::SetLearningRate(double learning_rate) {
    learning_rate_ = learning_rate;
}

double NeuralNetwork::GetLearningRate() const {
    return learning_rate_;
}

void NeuralNetwork::SetMomentum(double momentum) {
    momentum_ = momentum;
}

double NeuralNetwork::GetMomentum() const {
    return momentum_;
}

void NeuralNetwork::SetQpropMu(double mu) {
    qprop_mu_ = mu;
}

double NeuralNetwork::GetQpropMu() const {
    return qprop_mu_;
}

void NeuralNetwork::SetQpropWeightDecay(double weight_decay) {
    qprop_weight_decay_ = weight_decay;
}

double NeuralNetwork::GetQpropWeightDecay() const {
    return qprop_weight_decay_;
}

void NeuralNetwork::SetRpropWeightStepInitial(double weight_step) {
    rprop_weight_step_initial_ = weight_step;
}

double NeuralNetwork::GetRpropWeightStepInitial() const {
    return rprop_weight_step_initial_;
}

void NeuralNetwork::SetRpropWeightStepMin(double weight_step) {
    rprop_weight_step_min_ = weight_step;
}

double NeuralNetwork::GetRpropWeightStepMin() const {
    return rprop_weight_step_min_;
}

void NeuralNetwork::SetRpropWeightStepMax(double weight_step) {
    rprop_weight_step_max_ = weight_step;
}

double NeuralNetwork::GetRpropWeightStepMax() const {
    return rprop_weight_step_max_;
}

void NeuralNetwork::SetRpropIncreaseFactor(double factor) {
    rprop_increase_factor_ = factor;
}

double NeuralNetwork::GetRpropIncreaseFactor() const {
    return rprop_increase_factor_;
}

void NeuralNetwork::SetRpropDecreaseFactor(double factor) {
    rprop_decrease_factor_ = factor;
}

double NeuralNetwork::GetRpropDecreaseFactor() const {
    return rprop_decrease_factor_;
}

void NeuralNetwork::SetSarpropWeightDecayShift(double k1) {
    sarprop_weight_decay_shift_ = k1;
}

double NeuralNetwork::GetSarpropWeightDecayShift() const {
    return sarprop_weight_decay_shift_;
}

void NeuralNetwork::SetSarpropStepThresholdFactor(double k2) {
    sarprop_step_threshold_factor_ = k2;
}

double NeuralNetwork::GetSarpropStepThresholdFactor() const {
    return sarprop_step_threshold_factor_;
}

void NeuralNetwork::SetSarpropStepShift(double k3) {
    sarprop_step_shift_ = k3;
}

double NeuralNetwork::GetSarpropStepShift() const {
    return sarprop_step_shift_;
}

void NeuralNetwork::SetSarpropTemperature(double t) {
    sarprop_temperature_ = t;
}

double NeuralNetwork::GetSarpropTemperature() const {
    return sarprop_temperature_;
}

void NeuralNetwork::SetTrainingAlgorithmType(TrainingAlgorithmType type) {
    training_algorithm_type_ = type;
}

NeuralNetwork::TrainingAlgorithmType NeuralNetwork::GetTrainingAlgorithmType() const {
    return training_algorithm_type_;
}

void NeuralNetwork::SetErrorCostFunction(ErrorCostFunction mode) {
    error_cost_function_ = mode;
}

NeuralNetwork::ErrorCostFunction NeuralNetwork::GetErrorCostFunction() const {
    return error_cost_function_;
}

void NeuralNetwork::ResetWeightSteps() {
    const double initial_weight_step =
        (training_algorithm_type_ == TrainingAlgorithmType::ResilientBackpropagation ||
         training_algorithm_type_ == TrainingAlgorithmType::SimulatedAnnealingResilientBackpropagation) ?
        rprop_weight_step_initial_ :
        0;

    previous_weight_steps_.assign(weights_.size(), initial_weight_step);
}

void NeuralNetwork::ResetSlopes() {
    slopes_.assign(weights_.size(), 0);
}

void NeuralNetwork::ResetPreviousSlopes() {
    previous_slopes_.assign(weights_.size(), 0);
}

void NeuralNetwork::UpdateWeightsOnline() {
    for (size_t i = 0; i < GetInputConnectionCount(); i++) {
        const auto& connection = GetInputConnection(i);
        const auto& from_neuron = GetNeuron(connection.from_neuron_index);
        const auto& to_neuron = GetNeuron(connection.to_neuron_index);
        const double delta = -1.0 * learning_rate_ * to_neuron.error * from_neuron.value + momentum_ * previous_weight_steps_[i];
        previous_weight_steps_[i] = delta;
        weights_[i] += delta;
    }
}

void NeuralNetwork::UpdateWeightsOffline(size_t current_epoch, size_t step_count) {
    assert(step_count != 0);

    switch (training_algorithm_type_) {
    case TrainingAlgorithmType::BatchingBackpropagation:
        UpdateWeightsBatchingBackpropagation(step_count);
        break;
    case TrainingAlgorithmType::QuickBackpropagation:
        UpdateWeightsQuickBackpropagation(step_count);
        break;
    case TrainingAlgorithmType::ResilientBackpropagation:
        UpdateWeightsResilientBackpropagation();
        break;
    case TrainingAlgorithmType::SimulatedAnnealingResilientBackpropagation:
        UpdateWeightsSimulatedAnnealingResilientBackpropagation(current_epoch);
        break;
    default:
        assert(false);
    }
}

void NeuralNetwork::UpdateWeightsBatchingBackpropagation(size_t step_count) {
    const double epsilon = learning_rate_ / step_count;

    size_t weight_index = 0;
    for (auto& weight : weights_) {
        weight += slopes_[weight_index++] * epsilon;
    }
}

void NeuralNetwork::UpdateWeightsQuickBackpropagation(size_t step_count) {
    const double epsilon = learning_rate_ / step_count;
    const double shrink_factor = qprop_mu_ / (1.0 + qprop_mu_);

    size_t weight_index = 0;
    for (auto& weight : weights_) {
        const double previous_slope = previous_slopes_[weight_index];
        const double previous_weight_step = previous_weight_steps_[weight_index];
        const double current_slope = slopes_[weight_index] + qprop_weight_decay_ * weight;
        double weight_step = epsilon * current_slope;

        static constexpr double previous_weight_step_epsilon = 0.001;
        if (previous_weight_step > previous_weight_step_epsilon) {
            if (current_slope <= 0.0) {
                weight_step = 0.0;
            }

            if (current_slope > (shrink_factor * previous_slope)) {
                weight_step += qprop_mu_ * previous_weight_step;
            } else {
                weight_step += previous_weight_step * current_slope / (previous_slope - current_slope);
            }
        } else if (previous_weight_step < -previous_weight_step_epsilon) {
            if (current_slope >= 0.0) {
                weight_step = 0.0;
            }

            if (current_slope < (shrink_factor * previous_slope)) {
                weight_step += qprop_mu_ * previous_weight_step;
            } else {
                weight_step += previous_weight_step * current_slope / (previous_slope - current_slope);
            }
        }

        previous_slopes_[weight_index] = current_slope;
        previous_weight_steps_[weight_index] = weight_step;

        weight += weight_step;
        weight_index++;
    }
}

void NeuralNetwork::UpdateWeightsResilientBackpropagation() {
    size_t weight_index = 0;
    for (auto& weight : weights_) {
        const double previous_slope = previous_slopes_[weight_index];
        double current_slope = slopes_[weight_index];
        double weight_step = 0.0;
        const double previous_slope_times_current_slope = previous_slope * current_slope;
        const double previous_weight_step = std::max(previous_weight_steps_[weight_index], rprop_weight_step_min_);

        if (previous_slope_times_current_slope >= 0.0) {
            weight_step = std::min(previous_weight_step * rprop_increase_factor_, rprop_weight_step_max_);
        } else if (previous_slope_times_current_slope < 0.0) {
            weight_step = std::max(previous_weight_step * rprop_decrease_factor_, rprop_weight_step_min_);
            current_slope = 0.0;
        }

        previous_slopes_[weight_index] = current_slope;
        previous_weight_steps_[weight_index] = weight_step;

        const double weight_delta = std::signbit(current_slope) ? -1 * weight_step : weight_step;
        weight += weight_delta;
        weight_index++;
    }
}

void NeuralNetwork::UpdateWeightsSimulatedAnnealingResilientBackpropagation(size_t current_epoch) {
    size_t weight_index = 0;
    for (auto& weight : weights_) {
        const double previous_slope = previous_slopes_[weight_index];
        double current_slope = slopes_[weight_index] - sarprop_weight_decay_shift_ * weight * std::exp2(-sarprop_temperature_ * current_epoch);
        const double previous_slope_times_current_slope = previous_slope * current_slope;
        const double previous_weight_step = std::max(previous_weight_steps_[weight_index], rprop_weight_step_min_);
        double weight_step = 0.0;

        if (previous_slope_times_current_slope > 0.0) {
            weight_step = std::min(previous_weight_step * rprop_increase_factor_, rprop_weight_step_max_);
            const double weight_delta = std::signbit(current_slope) ? -1 * weight_step : weight_step;
            weight += weight_delta;
        } else if (previous_slope_times_current_slope < 0.0) {
            const double rms_error = std::sqrt(GetError());

            if (previous_weight_step < sarprop_step_threshold_factor_ * rms_error * rms_error) {
                weight_step = previous_weight_step * rprop_decrease_factor_ + sarprop_step_shift_ * random_.RandomFloat(0.0, 1.0) * rms_error * std::exp2(-sarprop_temperature_ * current_epoch);
            } else {
                weight_step = std::max(previous_weight_step * rprop_decrease_factor_, rprop_weight_step_min_);
            }

            current_slope = 0.0;
        } else {
            const double weight_delta = std::signbit(current_slope) ? -1 * previous_weight_step : previous_weight_step;
            weight += weight_delta;
        }

        previous_slopes_[weight_index] = current_slope;
        previous_weight_steps_[weight_index] = weight_step;
        weight_index++;
    }
}

void NeuralNetwork::UpdateSlopes() {
    for (size_t i = 0; i < GetInputConnectionCount(); i++) {
        const auto& connection = GetInputConnection(i);
        const auto& from_neuron = GetNeuron(connection.from_neuron_index);
        const auto& to_neuron = GetNeuron(connection.to_neuron_index);

        // Index into |slopes_| via the input connection index.
        slopes_[i] += -1.0 * from_neuron.value * to_neuron.error;
    }
}

void NeuralNetwork::TrainOffline(TrainingData* training_data, size_t epoch_count) {
    ResetPreviousSlopes();
    ResetWeightSteps();

    for (size_t i = 0; i < epoch_count; i++) {
        ResetSlopes();

        // Shuffle the training data each epoch.
        random_.ShuffleVector(training_data);

        // Train the network using offline weight updates - batching.
        for (const auto& example : *training_data) {
            // Run the network forward to get values in the output neurons.
            RunForward(example.input);

            // Run the network backward to propagate the error values.
            RunBackward(example.output);

            // Update slopes, but not weights - this is a batching algorithm.
            UpdateSlopes();
        }

        
        for (size_t j = 0; j < training_data->size(); j++) {
        }

        // Update weights.
        UpdateWeightsOffline(i, training_data->size());
    }
}

void NeuralNetwork::TrainOnline(TrainingData* training_data, size_t epoch_count) {
    ResetWeightSteps();

    for (size_t i = 0; i < epoch_count; i++) {
        // Shuffle the training data each epoch.
        random_.ShuffleVector(training_data);

        // Train the network using online weight updates - no batching.
        for (const auto& example : *training_data) {
            // Run the network forward to get values in the output neurons.
            RunForward(example.input);

            // Run the network backward to propagate the error values.
            RunBackward(example.output);

            // Update weights online - no batching.
            UpdateWeightsOnline();
        }
    }
}

void NeuralNetwork::Train(TrainingData* training_data, size_t epoch_count) {
    switch (training_algorithm_type_) {
    case TrainingAlgorithmType::Backpropagation:
        TrainOnline(training_data, epoch_count);
        break;
    case TrainingAlgorithmType::BatchingBackpropagation:
    case TrainingAlgorithmType::QuickBackpropagation:
    case TrainingAlgorithmType::ResilientBackpropagation:
    case TrainingAlgorithmType::SimulatedAnnealingResilientBackpropagation:
        TrainOffline(training_data, epoch_count);
        break;
    default:
        assert(false);
    }
}

void NeuralNetwork::ComputeNeuronValueRange(size_t neuron_start_index, size_t neuron_count) {
    for (size_t i = 0; i < neuron_count; i++) {
        ComputeNeuronValue(neuron_start_index + i);
    }
}

void NeuralNetwork::ComputeNeuronValue(size_t neuron_index) {
    auto& neuron = GetNeuron(neuron_index);
    neuron.field = 0.0;

    // Sum incoming values.
    for (size_t i = 0; i < neuron.input_connection_count; i++) {
        const size_t input_connection_index = neuron.input_connection_start_index + i;
        const auto& connection = GetInputConnection(input_connection_index);
        const auto& from_neuron = GetNeuron(connection.from_neuron_index);

        neuron.field += from_neuron.value * weights_[input_connection_index];
    }

    neuron.value = ExecuteActivationFunction(neuron.activation_function_type, neuron.value);
}

void NeuralNetwork::ComputeNeuronError(size_t neuron_index) {
    auto& neuron = GetNeuron(neuron_index);
    double sum = 0.0;

    // Sum outgoing errors.
    for (size_t i = 0; i < neuron.output_connection_count; i++) {
        const size_t output_connection_index = neuron.output_connection_start_index + i;
        const auto& output_connection = GetOutputConnection(output_connection_index);
        const auto& input_connection = GetInputConnection(output_connection.input_connection_index);
        const auto& to_neuron = GetNeuron(input_connection.to_neuron_index);

        sum += weights_[output_connection.input_connection_index] * to_neuron.error;
    }

    neuron.error = ExecuteActivationFunctionDerivative(neuron.activation_function_type, neuron.value, neuron.field) * sum;
}

void NeuralNetwork::RunForward(const std::vector<double>& input) {
    assert(is_constructed_);
    assert(input.size() == GetInputNeuronCount());

    // Feed each input into the corresponding input neuron.
    for (size_t i = 0; i < GetInputNeuronCount(); i++) {
        auto& neuron = GetInputNeuron(i);
        neuron.value = input[i];
    }

    // Pull the values from the input layer through the hidden layer neurons.
    ComputeNeuronValueRange(GetHiddenNeuronStartIndex(), GetHiddenNeuronCount());

    // Pull values into the output layer.
    ComputeNeuronValueRange(GetOutputNeuronStartIndex(), GetOutputNeuronCount());
}

void NeuralNetwork::RunBackward(const std::vector<double>& output) {
    assert(is_constructed_);
    assert(output.size() == GetOutputNeuronCount());

    ResetOutputLayerError();
    CalculateOutputLayerError(output);

    // Calculate error at each hidden layer neuron.
    const size_t hiddenNeuronStartIndex = GetHiddenNeuronStartIndex();
    for (size_t i = 0; i < GetHiddenNeuronCount(); i++) {
        ComputeNeuronError(hiddenNeuronStartIndex + (GetHiddenNeuronCount() - 1 - i));
    }
}

void NeuralNetwork::InitializeWeightsRandom(double min, double max) {
    assert(is_constructed_);

    for (auto& weight : weights_) {
        weight = random_.RandomFloat(min, max);
    }
}

void NeuralNetwork::InitializeWeights(const TrainingData& training_data) {
    assert(is_constructed_);
    assert(!training_data.empty());

    double min_input = std::numeric_limits<double>::max();
    double max_input = std::numeric_limits<double>::min();

    // Search all examples to find the min/max input values.
    for (const auto& example : training_data) {
        assert(GetInputNeuronCount() == example.input.size());
        const auto minmax = std::minmax_element(example.input.begin(), example.input.end());
        min_input = std::min(min_input, *minmax.first);
        max_input = std::max(max_input, *minmax.second);
    }

    constexpr double neuron_percentage = 0.7;
    const double factor = pow(neuron_percentage * GetHiddenNeuronCount(), 1.0 / GetHiddenNeuronCount()) / (max_input - min_input);
    InitializeWeightsRandom(-factor, factor);
}

double NeuralNetwork::GetError() const {
    assert(error_count_ != 0);

    return error_sum_ / error_count_;
}

double NeuralNetwork::GetError(const std::vector<double>& output) {
    ResetOutputLayerError();
    CalculateOutputLayerError(output);
    return GetError();
}

double NeuralNetwork::GetError(const TrainingData& training_data) {
    ResetOutputLayerError();

    for (const auto& example : training_data) {
        RunForward(example.input);
        CalculateOutputLayerError(example.output);
    }

    return GetError();
}

void NeuralNetwork::ResetOutputLayerError() {
    error_count_ = 0;
    error_sum_ = 0.0;
}

void NeuralNetwork::CalculateOutputLayerError(const std::vector<double>& output) {
    // Calculate error at each output neuron.
    const size_t output_neuron_start_index = GetOutputNeuronStartIndex();
    for (size_t i = 0; i < GetOutputNeuronCount(); i++) {
        auto& neuron = GetNeuron(output_neuron_start_index + i);
        const double delta = neuron.value - output[i];

        switch (this->error_cost_function_) {
        case ErrorCostFunction::MeanSquareError:
            this->error_sum_ += (delta * delta) / 2;
            break;
        case ErrorCostFunction::MeanAbsoluteError:
            this->error_sum_ += std::fabs(delta);
            break;
        }
        this->error_count_++;

        /*
        if (IsActivationFunctionSymmetric(neuron.activation_function_type)) {
            delta /= 2;
        }

        if (this->should_shape_error_curve_) {
            delta = ApplyErrorShaping(delta);
        }
        */

        neuron.error = ExecuteActivationFunctionDerivative(neuron.activation_function_type, neuron.value, neuron.field) * delta;
    }
}

std::vector<double>& NeuralNetwork::GetWeights() {
    return weights_;
}

void NeuralNetwork::SetWeights(const std::vector<double>& weights) {
    assert(weights_.size() == weights.size());
    weights_.assign(weights.cbegin(), weights.cend());
}

void NeuralNetwork::GetOutput(std::vector<double>* output) const {
    output->resize(GetOutputNeuronCount());
    for (size_t i = 0; i < GetOutputNeuronCount(); i++) {
        const auto& neuron = GetOutputNeuron(i);
        output->at(i) = neuron.value;
    }
}

}  // namespace panann
