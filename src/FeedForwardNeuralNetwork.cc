//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <cassert>

#include "FeedForwardNeuralNetwork.h"
#include "TrainingData.h"

namespace panann {

void FeedForwardNeuralNetwork::SetLearningRate(double learning_rate) {
    learning_rate_ = learning_rate;
}

double FeedForwardNeuralNetwork::GetLearningRate() const {
    return learning_rate_;
}

void FeedForwardNeuralNetwork::SetMomentum(double momentum) {
    momentum_ = momentum;
}

double FeedForwardNeuralNetwork::GetMomentum() const {
    return momentum_;
}

void FeedForwardNeuralNetwork::SetQpropMu(double mu) {
    qprop_mu_ = mu;
}

double FeedForwardNeuralNetwork::GetQpropMu() const {
    return qprop_mu_;
}

void FeedForwardNeuralNetwork::SetQpropWeightDecay(double weight_decay) {
    qprop_weight_decay_ = weight_decay;
}

double FeedForwardNeuralNetwork::GetQpropWeightDecay() const {
    return qprop_weight_decay_;
}

void FeedForwardNeuralNetwork::SetRpropWeightStepInitial(double weight_step) {
    rprop_weight_step_initial_ = weight_step;
}

double FeedForwardNeuralNetwork::GetRpropWeightStepInitial() const {
    return rprop_weight_step_initial_;
}

void FeedForwardNeuralNetwork::SetRpropWeightStepMin(double weight_step) {
    rprop_weight_step_min_ = weight_step;
}

double FeedForwardNeuralNetwork::GetRpropWeightStepMin() const {
    return rprop_weight_step_min_;
}

void FeedForwardNeuralNetwork::SetRpropWeightStepMax(double weight_step) {
    rprop_weight_step_max_ = weight_step;
}

double FeedForwardNeuralNetwork::GetRpropWeightStepMax() const {
    return rprop_weight_step_max_;
}

void FeedForwardNeuralNetwork::SetRpropIncreaseFactor(double factor) {
    rprop_increase_factor_ = factor;
}

double FeedForwardNeuralNetwork::GetRpropIncreaseFactor() const {
    return rprop_increase_factor_;
}

void FeedForwardNeuralNetwork::SetRpropDecreaseFactor(double factor) {
    rprop_decrease_factor_ = factor;
}

double FeedForwardNeuralNetwork::GetRpropDecreaseFactor() const {
    return rprop_decrease_factor_;
}

void FeedForwardNeuralNetwork::SetSarpropWeightDecayShift(double k1) {
    sarprop_weight_decay_shift_ = k1;
}

double FeedForwardNeuralNetwork::GetSarpropWeightDecayShift() const {
    return sarprop_weight_decay_shift_;
}

void FeedForwardNeuralNetwork::SetSarpropStepThresholdFactor(double k2) {
    sarprop_step_threshold_factor_ = k2;
}

double FeedForwardNeuralNetwork::GetSarpropStepThresholdFactor() const {
    return sarprop_step_threshold_factor_;
}

void FeedForwardNeuralNetwork::SetSarpropStepShift(double k3) {
    sarprop_step_shift_ = k3;
}

double FeedForwardNeuralNetwork::GetSarpropStepShift() const {
    return sarprop_step_shift_;
}

void FeedForwardNeuralNetwork::SetSarpropTemperature(double t) {
    sarprop_temperature_ = t;
}

double FeedForwardNeuralNetwork::GetSarpropTemperature() const {
    return sarprop_temperature_;
}

void FeedForwardNeuralNetwork::SetTrainingAlgorithmType(TrainingAlgorithmType type) {
    training_algorithm_type_ = type;
}

FeedForwardNeuralNetwork::TrainingAlgorithmType FeedForwardNeuralNetwork::GetTrainingAlgorithmType() const {
    return training_algorithm_type_;
}

void FeedForwardNeuralNetwork::ResetWeightSteps() {
    const double initial_weight_step =
        (training_algorithm_type_ == TrainingAlgorithmType::ResilientBackpropagation ||
         training_algorithm_type_ == TrainingAlgorithmType::SimulatedAnnealingResilientBackpropagation) ?
        rprop_weight_step_initial_ :
        0;

    previous_weight_steps_.assign(GetWeightCount(), initial_weight_step);
}

void FeedForwardNeuralNetwork::ResetSlopes() {
    slopes_.assign(GetWeightCount(), 0);
}

void FeedForwardNeuralNetwork::ResetPreviousSlopes() {
    previous_slopes_.assign(GetWeightCount(), 0);
}

void FeedForwardNeuralNetwork::UpdateWeightsOnline() {
    for (size_t i = 0; i < GetInputConnectionCount(); i++) {
        const auto& connection = GetInputConnection(i);
        const auto& from_neuron = GetNeuron(connection.from_neuron_index);
        const auto& to_neuron = GetNeuron(connection.to_neuron_index);
        const double delta = -1.0 * learning_rate_ * to_neuron.error * from_neuron.value + momentum_ * previous_weight_steps_[i];
        previous_weight_steps_[i] = delta;
        GetWeight(i) += delta;
    }
}

void FeedForwardNeuralNetwork::UpdateWeightsOffline(size_t current_epoch, size_t step_count) {
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

void FeedForwardNeuralNetwork::UpdateWeightsBatchingBackpropagation(size_t step_count) {
    const double epsilon = learning_rate_ / step_count;
    for (size_t i = 0; i < GetWeightCount(); i++) {
        GetWeight(i) += slopes_[i] * epsilon;
    }
}

void FeedForwardNeuralNetwork::UpdateWeightsQuickBackpropagation(size_t step_count) {
    const double epsilon = learning_rate_ / step_count;
    const double shrink_factor = qprop_mu_ / (1.0 + qprop_mu_);

    for (size_t i = 0; i < GetWeightCount(); i++) {
        auto& weight = GetWeight(i);
        const double previous_slope = previous_slopes_[i];
        const double previous_weight_step = previous_weight_steps_[i];
        const double current_slope = slopes_[i] + qprop_weight_decay_ * weight;
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

        previous_slopes_[i] = current_slope;
        previous_weight_steps_[i] = weight_step;
        weight += weight_step;
    }
}

void FeedForwardNeuralNetwork::UpdateWeightsResilientBackpropagation() {
    for (size_t i = 0; i < GetWeightCount(); i++) {
        auto& weight = GetWeight(i);
        const double previous_slope = previous_slopes_[i];
        double current_slope = slopes_[i];
        double weight_step = 0.0;
        const double previous_slope_times_current_slope = previous_slope * current_slope;
        const double previous_weight_step = std::max(previous_weight_steps_[i], rprop_weight_step_min_);

        if (previous_slope_times_current_slope >= 0.0) {
            weight_step = std::min(previous_weight_step * rprop_increase_factor_, rprop_weight_step_max_);
        } else if (previous_slope_times_current_slope < 0.0) {
            weight_step = std::max(previous_weight_step * rprop_decrease_factor_, rprop_weight_step_min_);
            current_slope = 0.0;
        }

        previous_slopes_[i] = current_slope;
        previous_weight_steps_[i] = weight_step;

        const double weight_delta = std::signbit(current_slope) ? -1 * weight_step : weight_step;
        weight += weight_delta;
    }
}

void FeedForwardNeuralNetwork::UpdateWeightsSimulatedAnnealingResilientBackpropagation(size_t current_epoch) {
    for (size_t i = 0; i < GetWeightCount(); i++) {
        auto& weight = GetWeight(i);
        const double previous_slope = previous_slopes_[i];
        double current_slope = slopes_[i] - sarprop_weight_decay_shift_ * weight * std::exp2(-sarprop_temperature_ * current_epoch);
        const double previous_slope_times_current_slope = previous_slope * current_slope;
        const double previous_weight_step = std::max(previous_weight_steps_[i], rprop_weight_step_min_);
        double weight_step = 0.0;

        if (previous_slope_times_current_slope > 0.0) {
            weight_step = std::min(previous_weight_step * rprop_increase_factor_, rprop_weight_step_max_);
            const double weight_delta = std::signbit(current_slope) ? -1 * weight_step : weight_step;
            weight += weight_delta;
        } else if (previous_slope_times_current_slope < 0.0) {
            const double rms_error = std::sqrt(GetError());

            if (previous_weight_step < sarprop_step_threshold_factor_ * rms_error * rms_error) {
                weight_step = previous_weight_step * rprop_decrease_factor_ + sarprop_step_shift_ * GetRandom().RandomFloat(0.0, 1.0) * rms_error * std::exp2(-sarprop_temperature_ * current_epoch);
            } else {
                weight_step = std::max(previous_weight_step * rprop_decrease_factor_, rprop_weight_step_min_);
            }

            current_slope = 0.0;
        } else {
            const double weight_delta = std::signbit(current_slope) ? -1 * previous_weight_step : previous_weight_step;
            weight += weight_delta;
        }

        previous_slopes_[i] = current_slope;
        previous_weight_steps_[i] = weight_step;
    }
}

void FeedForwardNeuralNetwork::UpdateSlopes() {
    for (size_t i = 0; i < GetInputConnectionCount(); i++) {
        const auto& connection = GetInputConnection(i);
        const auto& from_neuron = GetNeuron(connection.from_neuron_index);
        const auto& to_neuron = GetNeuron(connection.to_neuron_index);

        // Index into |slopes_| via the input connection index.
        slopes_[i] += -1.0 * from_neuron.value * to_neuron.error;
    }
}

void FeedForwardNeuralNetwork::TrainOffline(TrainingData* training_data, size_t epoch_count) {
    ResetPreviousSlopes();
    ResetWeightSteps();

    for (size_t i = 0; i < epoch_count; i++) {
        ResetSlopes();

        // Shuffle the training data each epoch.
        GetRandom().ShuffleVector(training_data);

        // Train the network using offline weight updates - batching.
        for (const auto& example : *training_data) {
            // Run the network forward to get values in the output neurons.
            RunForward(example.input);

            // Run the network backward to propagate the error values.
            RunBackward(example.output);

            // Update slopes, but not weights - this is a batching algorithm.
            UpdateSlopes();
        }

        // Update weights.
        UpdateWeightsOffline(i, training_data->size());
    }
}

void FeedForwardNeuralNetwork::TrainOnline(TrainingData* training_data, size_t epoch_count) {
    ResetWeightSteps();

    for (size_t i = 0; i < epoch_count; i++) {
        // Shuffle the training data each epoch.
        GetRandom().ShuffleVector(training_data);

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

void FeedForwardNeuralNetwork::Train(TrainingData* training_data, size_t epoch_count) {
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

}  // namespace panann
