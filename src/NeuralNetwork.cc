//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <cassert>

#include "ActivationFunction.h"
#include "NeuralNetwork.h"
#include "TrainingData.h"

namespace panann {

size_t NeuralNetwork::GetInputNeuronCount() const {
    return input_neuron_count_;
}

void NeuralNetwork::SetInputNeuronCount(size_t input_neuron_count) {
    assert(!is_constructed_);
    input_neuron_count_ = input_neuron_count;
}

size_t NeuralNetwork::GetOutputNeuronCount() const {
    return output_neuron_count_;
}

void NeuralNetwork::SetOutputNeuronCount(size_t output_neuron_count) {
    assert(!is_constructed_);
    output_neuron_count_ = output_neuron_count;
}

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

void NeuralNetwork::SetQpropWeightDecay(double weightDecay) {
    qprop_weight_decay_ = weightDecay;
}

double NeuralNetwork::GetQpropWeightDecay() const {
    return qprop_weight_decay_;
}

void NeuralNetwork::SetRpropWeightStepInitial(double weightStep) {
    rprop_weight_step_initial_ = weightStep;
}

double NeuralNetwork::GetRpropWeightStepInitial() const {
    return rprop_weight_step_initial_;
}

void NeuralNetwork::SetRpropWeightStepMin(double weightStep) {
    rprop_weight_step_min_ = weightStep;
}

double NeuralNetwork::GetRpropWeightStepMin() const {
    return rprop_weight_step_min_;
}

void NeuralNetwork::SetRpropWeightStepMax(double weightStep) {
    rprop_weight_step_max_ = weightStep;
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

void NeuralNetwork::EnableShortcutConnections() {
    assert(!is_constructed_);
    enable_shortcut_connections_ = true;
}

void NeuralNetwork::DisableShortcutConnections() {
    assert(!is_constructed_);
    enable_shortcut_connections_ = false;
}

void NeuralNetwork::SetHiddenNeuronActivationFunctionType(ActivationFunctionType type) {
    assert(!is_constructed_);
    hidden_neuron_activation_function_type_ = type;
}

NeuralNetwork::ActivationFunctionType NeuralNetwork::GetHiddenNeuronActivationFunctionType() const {
    return hidden_neuron_activation_function_type_;
}

void NeuralNetwork::SetOutputNeuronActivationFunctionType(ActivationFunctionType type) {
    assert(!is_constructed_);
    output_neuron_activation_function_type_ = type;
}

NeuralNetwork::ActivationFunctionType NeuralNetwork::GetOutputNeuronActivationFunctionType() const {
    return output_neuron_activation_function_type_;
}

void NeuralNetwork::SetNeuronActivationFunction(size_t neuronIndex, ActivationFunctionType type) {
    assert(is_constructed_);
    assert(neuronIndex < neurons_.size());
    neurons_[neuronIndex].activation_function_type = type;
}

void NeuralNetwork::SetErrorCostFunction(ErrorCostFunction mode) {
    error_cost_function_ = mode;
}

NeuralNetwork::ErrorCostFunction NeuralNetwork::GetErrorCostFunction() const {
    return error_cost_function_;
}

/**
 * | hidden neurons | input neurons | output neurons | bias neurons |
 */
size_t NeuralNetwork::GetInputNeuronStartIndex() const {
    return hidden_neuron_count_;
}

size_t NeuralNetwork::GetOutputNeuronStartIndex() const {
    return hidden_neuron_count_ + this->input_neuron_count_;
}

size_t NeuralNetwork::GetHiddenNeuronStartIndex() const {
    return 0;
}

size_t NeuralNetwork::GetHiddenNeuronCount() const {
    return hidden_neuron_count_;
}

size_t NeuralNetwork::GetBiasNeuronStartIndex() const {
    return input_neuron_count_ + output_neuron_count_ + hidden_neuron_count_;
}

void NeuralNetwork::AddHiddenLayer(size_t neuron_count) {
    assert(!is_constructed_);

    // Add a new hidden layer beginning at the current hidden neuron count and continuing for |neuron_count| neurons.
    hidden_layers_.push_back({GetHiddenNeuronStartIndex() + hidden_neuron_count_, neuron_count});
    hidden_neuron_count_ += neuron_count;
}

void NeuralNetwork::Allocate() {
    assert(!is_constructed_);
    // Do not support networks with no hidden layers.
    assert(!hidden_layers_.empty());

    // Total count of neurons is all the input, output, and hidden neurons.
    // The input layer and hidden layers also contribute one bias neuron each.
    size_t neuron_count =
        input_neuron_count_ + 1 +
        output_neuron_count_ +
        hidden_neuron_count_ + hidden_layers_.size();

    neurons_.resize(neuron_count);

    size_t bias_neuron_index = GetBiasNeuronStartIndex();
    size_t input_connection_index = 0;
    size_t outputConnectionIndex = 0;
    size_t currentLayerInputConnectionCount = 0;
    size_t currentLayerOutputConnectionCount = 0;

    // Calculate the connections outgoing from the input layer.
    if (this->enable_shortcut_connections_) {
        // The input layer connects to all hidden layers and the output layer.
        currentLayerOutputConnectionCount = this->hidden_neuron_count_ + this->output_neuron_count_;
    } else {
        // The input layer connects only to the first hidden layer.
        currentLayerOutputConnectionCount = this->hidden_layers_.front().neuron_count;
    }

    size_t inputNeuronIndex = this->GetInputNeuronStartIndex();
    for (size_t i = 0; i < this->input_neuron_count_; i++) {
        Neuron& neuron = this->neurons_[inputNeuronIndex + i];
        neuron.output_connection_start_index = outputConnectionIndex;
        outputConnectionIndex += currentLayerOutputConnectionCount;
    }

    // The first bias neuron is the one for the input layer.
    Neuron& biasNeuronInput = this->neurons_[bias_neuron_index++];
    biasNeuronInput.output_connection_start_index = outputConnectionIndex;
    biasNeuronInput.value = 1.0;
    outputConnectionIndex += this->hidden_layers_.front().neuron_count;

    // Calculate the connections incoming to the output layer.
    if (this->enable_shortcut_connections_) {
        // All input and hidden neurons are connected to each output neuron.
        currentLayerInputConnectionCount = this->input_neuron_count_ + this->hidden_neuron_count_ + 1;
    } else {
        // Output neurons are connected only to the last hidden layer.
        currentLayerInputConnectionCount = this->hidden_layers_.back().neuron_count + 1;
    }

    size_t firstOutputNeuronIndex = this->GetOutputNeuronStartIndex();
    for (size_t i = 0; i < this->output_neuron_count_; i++) {
        Neuron& neuron = this->neurons_[firstOutputNeuronIndex + i];
        neuron.input_connection_start_index = input_connection_index;
        neuron.activation_function_type = this->output_neuron_activation_function_type_;
        input_connection_index += currentLayerInputConnectionCount;
    }

    size_t neuronIndex = this->GetHiddenNeuronStartIndex();

    // Calculate the connections to and from all hidden layers.
    for (size_t layerIndex = 0; layerIndex < this->hidden_layers_.size(); layerIndex++) {
        currentLayerInputConnectionCount = 0;
        currentLayerOutputConnectionCount = 0;

        if (this->enable_shortcut_connections_) {
            // All hidden layers connect to the input layer when shortcuts are enabled.
            currentLayerInputConnectionCount += this->input_neuron_count_ + 1;

            // Each neuron in this layer connects to the neurons in all previous hidden layers.
            for (size_t previousLayerIndex = 0; previousLayerIndex < layerIndex; previousLayerIndex++) {
                currentLayerInputConnectionCount += this->hidden_layers_[previousLayerIndex].neuron_count + 1;
            }

            // All hidden layers connect directly to the output layer when shortcuts are enabled.
            currentLayerOutputConnectionCount += this->output_neuron_count_;

            // This layer connects to all neurons in subsequent hidden layers.
            for (size_t nextLayerIndex = layerIndex + 1; nextLayerIndex < this->hidden_layers_.size(); nextLayerIndex++) {
                currentLayerOutputConnectionCount += this->hidden_layers_[nextLayerIndex].neuron_count;
            }
        } else {
            if (layerIndex == 0) {
                // First hidden layer connects to the input layer.
                currentLayerInputConnectionCount += this->input_neuron_count_ + 1;
            } else {
                // This hidden layer connects directly the previous one.
                currentLayerInputConnectionCount += this->hidden_layers_[layerIndex - 1].neuron_count + 1;
            }

            if (layerIndex == this->hidden_layers_.size() - 1) {
                // Last hidden layer connects to the output layer.
                currentLayerOutputConnectionCount += this->output_neuron_count_;
            } else {
                assert(layerIndex + 1 < this->hidden_layers_.size());

                // This hidden layer connects directly to the next one.
                currentLayerOutputConnectionCount += this->hidden_layers_[layerIndex + 1].neuron_count;
            }
        }

        const Layer& currentLayer = this->hidden_layers_[layerIndex];
        for (size_t i = 0; i < currentLayer.neuron_count; i++) {
            Neuron& neuron = this->neurons_[neuronIndex++];
            neuron.input_connection_start_index = input_connection_index;
            neuron.output_connection_start_index = outputConnectionIndex;
            neuron.activation_function_type = this->hidden_neuron_activation_function_type_;

            input_connection_index += currentLayerInputConnectionCount;
            outputConnectionIndex += currentLayerOutputConnectionCount;
        }

        // Bias neurons cannot have shortcut connections.
        size_t biasOutputConnections = 0;
        if (layerIndex == this->hidden_layers_.size() - 1) {
            // Bias neuron in the last hidden layer connects to the output layer.
            biasOutputConnections = this->output_neuron_count_;
        } else {
            // Bias neuron in this hidden layer connects to the next hidden layer.
            biasOutputConnections = this->hidden_layers_[layerIndex + 1].neuron_count;
        }

        // Bias neurons do not have incoming connections.
        Neuron& biasNeuron = this->neurons_[bias_neuron_index++];
        biasNeuron.output_connection_start_index = outputConnectionIndex;
        biasNeuron.value = 1.0;
        outputConnectionIndex += biasOutputConnections;
    }

    this->input_connections_.resize(input_connection_index);
    this->output_connections_.resize(outputConnectionIndex);
    this->weights_.resize(input_connection_index);
}

void NeuralNetwork::ConnectNeurons(size_t fromNeuronIndex, size_t toNeuronIndex) {
    Neuron& fromNeuron = this->neurons_[fromNeuronIndex];
    Neuron& toNeuron = this->neurons_[toNeuronIndex];

    size_t input_connection_index = toNeuron.input_connection_start_index + toNeuron.input_connection_count;
    InputConnection& inputConnection = this->input_connections_.at(input_connection_index);
    inputConnection.from_neuron_index = fromNeuronIndex;
    inputConnection.to_neuron_index = toNeuronIndex;
    toNeuron.input_connection_count++;

    size_t outputConnectionIndex = fromNeuron.output_connection_start_index + fromNeuron.output_connection_count;
    OutputConnection& outputConnection = this->output_connections_.at(outputConnectionIndex);
    outputConnection.input_connection_index = input_connection_index;
    fromNeuron.output_connection_count++;
}

void NeuralNetwork::ConnectLayerToNeuron(size_t fromNeuronIndex, size_t fromNeuronCount, size_t toNeuronIndex) {
    for (size_t i = 0; i < fromNeuronCount; i++) {
        ConnectNeurons(fromNeuronIndex + i, toNeuronIndex);
    }
}

void NeuralNetwork::ConnectLayers(size_t fromNeuronIndex, size_t fromNeuronCount, size_t toNeuronIndex, size_t toNeuronCount) {
    for (size_t i = 0; i < toNeuronCount; i++) {
        ConnectLayerToNeuron(fromNeuronIndex, fromNeuronCount, toNeuronIndex + i);
    }
}

void NeuralNetwork::ConnectBiasNeuron(size_t bias_neuron_index, size_t toNeuronIndex, size_t toNeuronCount) {
    ConnectLayers(bias_neuron_index, 1, toNeuronIndex, toNeuronCount);
}

void NeuralNetwork::ConnectFully() {
    assert(!this->is_constructed_);
    assert(!this->hidden_layers_.empty());

    size_t inputNeuronStartIndex = this->GetInputNeuronStartIndex();
    size_t bias_neuron_index = this->GetBiasNeuronStartIndex();

    for (size_t layerIndex = 0; layerIndex < this->hidden_layers_.size(); layerIndex++) {
        const Layer& currentLayer = this->hidden_layers_[layerIndex];

        if (this->enable_shortcut_connections_) {
            // Connect to input layer.
            ConnectLayers(inputNeuronStartIndex,
                          this->input_neuron_count_,
                          currentLayer.neuron_start_index,
                          currentLayer.neuron_count);

            // Connect to all previous hidden layers.
            for (size_t previousLayerIndex = 0; previousLayerIndex < layerIndex; previousLayerIndex++) {
                const Layer& previousLayer = this->hidden_layers_[previousLayerIndex];
                ConnectLayers(previousLayer.neuron_start_index,
                              previousLayer.neuron_count,
                              currentLayer.neuron_start_index,
                              currentLayer.neuron_count);
            }
        } else {
            if (layerIndex == 0) {
                // Connect first hidden layer to input layer.
                ConnectLayers(inputNeuronStartIndex,
                              this->input_neuron_count_,
                              currentLayer.neuron_start_index,
                              currentLayer.neuron_count);
            } else {
                // Connect to previous hidden layer.
                const Layer& previousLayer = this->hidden_layers_[layerIndex - 1];
                ConnectLayers(previousLayer.neuron_start_index,
                              previousLayer.neuron_count,
                              currentLayer.neuron_start_index,
                              currentLayer.neuron_count);
            }
        }

        // Bias neurons do not have shortcut connections.
        // Just connect this layer to the bias neuron in the layer before it.
        ConnectBiasNeuron(bias_neuron_index++, currentLayer.neuron_start_index, currentLayer.neuron_count);
    }

    size_t outputNeuronStartIndex = this->GetOutputNeuronStartIndex();
    if (this->enable_shortcut_connections_) {
        // Connect input layer to output layer.
        ConnectLayers(inputNeuronStartIndex,
                      this->input_neuron_count_,
                      outputNeuronStartIndex,
                      this->output_neuron_count_);

        // Connect all hidden layers to output layer.
        for (size_t i = 0; i < this->hidden_layers_.size(); i++) {
            const Layer& layer = this->hidden_layers_[i];
            ConnectLayers(layer.neuron_start_index,
                          layer.neuron_count,
                          outputNeuronStartIndex,
                          this->output_neuron_count_);
        }
    } else {
        const Layer& previousLayer = this->hidden_layers_.back();
        // Connect output layer to the last hidden layer.
        ConnectLayers(previousLayer.neuron_start_index,
                      previousLayer.neuron_count,
                      outputNeuronStartIndex,
                      this->output_neuron_count_);
    }

    // Connect output layer to the bias neuron in the last hidden layer.
    ConnectBiasNeuron(bias_neuron_index, outputNeuronStartIndex, this->output_neuron_count_);
}

void NeuralNetwork::Construct() {
    assert(!this->is_constructed_);

    this->Allocate();
    this->ConnectFully();

    this->is_constructed_ = true;
}

void NeuralNetwork::ResetWeightSteps() {
    double initialWeightStep =
        (this->training_algorithm_type_ == TrainingAlgorithmType::ResilientBackpropagation ||
         this->training_algorithm_type_ == TrainingAlgorithmType::SimulatedAnnealingResilientBackpropagation) ?
        this->rprop_weight_step_initial_ :
        0;

    this->previous_weight_steps_.resize(this->weights_.size());
    std::fill(this->previous_weight_steps_.begin(), this->previous_weight_steps_.end(), initialWeightStep);
}

void NeuralNetwork::ResetSlopes() {
    this->slopes_.resize(this->weights_.size());
    std::fill(this->slopes_.begin(), this->slopes_.end(), 0);
}

void NeuralNetwork::ResetPreviousSlopes() {
    this->previous_slopes_.resize(this->weights_.size());
    std::fill(this->previous_slopes_.begin(), this->previous_slopes_.end(), 0);
}

void NeuralNetwork::UpdateWeightsOnline() {
    for (size_t i = 0; i < this->input_connections_.size(); i++) {
        const InputConnection& connection = this->input_connections_[i];
        const Neuron& fromNeuron = this->neurons_[connection.from_neuron_index];
        const Neuron& toNeuron = this->neurons_[connection.to_neuron_index];

        double delta = -1.0 * this->learning_rate_ * toNeuron.error * fromNeuron.value + this->momentum_ * this->previous_weight_steps_[i];
        this->previous_weight_steps_[i] = delta;
        this->weights_[i] += delta;
    }
}

void NeuralNetwork::UpdateWeightsOffline(size_t currentEpoch, size_t stepCount) {
    assert(stepCount != 0);

    switch (this->training_algorithm_type_) {
    case TrainingAlgorithmType::BatchingBackpropagation:
        this->UpdateWeightsBatchingBackpropagation(stepCount);
        break;
    case TrainingAlgorithmType::QuickBackpropagation:
        this->UpdateWeightsQuickBackpropagation(stepCount);
        break;
    case TrainingAlgorithmType::ResilientBackpropagation:
        this->UpdateWeightsResilientBackpropagation();
        break;
    case TrainingAlgorithmType::SimulatedAnnealingResilientBackpropagation:
        this->UpdateWeightsSimulatedAnnealingResilientBackpropagation(currentEpoch);
        break;
    default:
        assert(false);
    }
}

void NeuralNetwork::UpdateWeightsBatchingBackpropagation(size_t stepCount) {
    double epsilon = this->learning_rate_ / stepCount;

    for (size_t i = 0; i < this->weights_.size(); i++) {
        this->weights_[i] += this->slopes_[i] * epsilon;
    }
}

void NeuralNetwork::UpdateWeightsQuickBackpropagation(size_t stepCount) {
    double epsilon = this->learning_rate_ / stepCount;
    double shrinkFactor = this->qprop_mu_ / (1.0 + this->qprop_mu_);

    for (size_t i = 0; i < this->weights_.size(); i++) {
        double previousSlope = this->previous_slopes_[i];
        double previousWeightStep = this->previous_weight_steps_[i];
        double currentSlope = this->slopes_[i] + this->qprop_weight_decay_ * this->weights_[i];
        double weightStep = epsilon * currentSlope;

        if (previousWeightStep > 0.001) {
            if (currentSlope <= 0.0) {
                weightStep = 0.0;
            }

            if (currentSlope > (shrinkFactor * previousSlope)) {
                weightStep += this->qprop_mu_ * previousWeightStep;
            } else {
                weightStep += previousWeightStep * currentSlope / (previousSlope - currentSlope);
            }
        } else if (previousWeightStep < -0.001) {
            if (currentSlope >= 0.0) {
                weightStep = 0.0;
            }

            if (currentSlope < (shrinkFactor * previousSlope)) {
                weightStep += this->qprop_mu_ * previousWeightStep;
            } else {
                weightStep += previousWeightStep * currentSlope / (previousSlope - currentSlope);
            }
        }

        this->previous_slopes_[i] = currentSlope;
        this->previous_weight_steps_[i] = weightStep;

        this->weights_[i] += weightStep;
    }
}

void NeuralNetwork::UpdateWeightsResilientBackpropagation() {
    for (size_t i = 0; i < this->weights_.size(); i++) {
        double previousSlope = this->previous_slopes_[i];
        double currentSlope = this->slopes_[i];
        double weightStep = 0.0;
        double previousSlopeTimesCurrentSlope = previousSlope * currentSlope;
        double previousWeightStep = std::max(this->previous_weight_steps_[i], this->rprop_weight_step_min_);

        if (previousSlopeTimesCurrentSlope >= 0.0) {
            weightStep = std::min(previousWeightStep * this->rprop_increase_factor_, this->rprop_weight_step_max_);
        } else if (previousSlopeTimesCurrentSlope < 0.0) {
            weightStep = std::max(previousWeightStep * this->rprop_decrease_factor_, this->rprop_weight_step_min_);
            currentSlope = 0.0;
        }

        double weightDelta = std::signbit(currentSlope) ? -1 * weightStep : weightStep;
        this->weights_[i] += weightDelta;

        this->previous_slopes_[i] = currentSlope;
        this->previous_weight_steps_[i] = weightStep;
    }
}

void NeuralNetwork::UpdateWeightsSimulatedAnnealingResilientBackpropagation(size_t currentEpoch) {
    for (size_t i = 0; i < this->weights_.size(); i++) {
        double previousSlope = this->previous_slopes_[i];
        double currentSlope = this->slopes_[i] - this->sarprop_weight_decay_shift_ * this->weights_[i] * std::exp2(-this->sarprop_temperature_ * currentEpoch);
        double previousSlopeTimesCurrentSlope = previousSlope * currentSlope;
        double previousWeightStep = std::max(this->previous_weight_steps_[i], this->rprop_weight_step_min_);
        double weightStep = 0.0;

        if (previousSlopeTimesCurrentSlope > 0.0) {
            weightStep = std::min(previousWeightStep * this->rprop_increase_factor_, this->rprop_weight_step_max_);
            double weightDelta = std::signbit(currentSlope) ? -1 * weightStep : weightStep;
            this->weights_[i] += weightDelta;
        } else if (previousSlopeTimesCurrentSlope < 0.0) {
            double rmsError = std::sqrt(this->GetError());

            if (previousWeightStep < this->sarprop_step_threshold_factor_ * rmsError * rmsError) {
                weightStep = previousWeightStep * this->rprop_decrease_factor_ + this->sarprop_step_shift_ * this->random_.RandomFloat(0.0, 1.0) * rmsError * std::exp2(-this->sarprop_temperature_ * currentEpoch);
            } else {
                weightStep = std::max(previousWeightStep * this->rprop_decrease_factor_, this->rprop_weight_step_min_);
            }

            currentSlope = 0.0;
        } else {
            double weightDelta = std::signbit(currentSlope) ? -1 * previousWeightStep : previousWeightStep;
            this->weights_[i] += weightDelta;
        }

        this->previous_slopes_[i] = currentSlope;
        this->previous_weight_steps_[i] = weightStep;
    }
}

void NeuralNetwork::UpdateSlopes() {
    for (size_t i = 0; i < this->input_connections_.size(); i++) {
        const InputConnection& connection = this->input_connections_[i];
        const Neuron& fromNeuron = this->neurons_[connection.from_neuron_index];
        const Neuron& toNeuron = this->neurons_[connection.to_neuron_index];

        this->slopes_[i] += -1.0 * fromNeuron.value * toNeuron.error;
    }
}

void NeuralNetwork::TrainOffline(TrainingData* trainingData, size_t epochCount) {
    this->ResetPreviousSlopes();
    this->ResetWeightSteps();

    for (size_t i = 0; i < epochCount; i++) {
        this->random_.ShuffleVector(trainingData);
        this->ResetSlopes();

        // Train the network using offline weight updates - batching
        for (size_t j = 0; j < trainingData->size(); j++) {
            // Run the network forward to get values in the output neurons.
            this->RunForward(&trainingData->operator[](j).input);

            // Run the network backward to propagate the error values
            this->RunBackward(&trainingData->operator[](j).output);

            // Update slopes, but not weights - this is a batching algorithm
            this->UpdateSlopes();
        }

        // Update weights
        this->UpdateWeightsOffline(i, trainingData->size());
    }
}

void NeuralNetwork::TrainOnline(TrainingData* trainingData, size_t epochCount) {
    this->ResetWeightSteps();

    for (size_t i = 0; i < epochCount; i++) {
        this->random_.ShuffleVector(trainingData);

        // Train the network using online weight updates - no batching
        for (size_t j = 0; j < trainingData->size(); j++) {
            // Run the network forward to get values in the output neurons.
            this->RunForward(&trainingData->operator[](j).input);

            // Run the network backward to propagate the error values
            this->RunBackward(&trainingData->operator[](j).output);

            // Update weights online - no batching
            this->UpdateWeightsOnline();
        }
    }
}

void NeuralNetwork::Train(TrainingData* trainingData, size_t epochCount) {
    switch (this->training_algorithm_type_) {
    case TrainingAlgorithmType::Backpropagation:
        this->TrainOnline(trainingData, epochCount);
        break;
    case TrainingAlgorithmType::BatchingBackpropagation:
    case TrainingAlgorithmType::QuickBackpropagation:
    case TrainingAlgorithmType::ResilientBackpropagation:
    case TrainingAlgorithmType::SimulatedAnnealingResilientBackpropagation:
        this->TrainOffline(trainingData, epochCount);
        break;
    default:
        assert(false);
    }
}

void NeuralNetwork::ComputeNeuronValueRange(size_t neuronStartIndex, size_t neuron_count) {
    for (size_t i = 0; i < neuron_count; i++) {
        this->ComputeNeuronValue(neuronStartIndex + i);
    }
}

void NeuralNetwork::ComputeNeuronValue(size_t neuronIndex) {
    Neuron& neuron = this->neurons_[neuronIndex];
    neuron.field = 0.0;

    // Sum incoming values.
    for (size_t i = 0; i < neuron.input_connection_count; i++) {
        size_t input_connection_index = neuron.input_connection_start_index + i;
        const InputConnection& connection = this->input_connections_[input_connection_index];
        const Neuron& fromNeuron = this->neurons_[connection.from_neuron_index];

        neuron.field += fromNeuron.value * this->weights_[input_connection_index];
    }

    neuron.value = ExecuteActivationFunction(&neuron);
}

void NeuralNetwork::ComputeNeuronError(size_t neuronIndex) {
    Neuron& neuron = this->neurons_[neuronIndex];
    double sum = 0.0;

    // Sum outgoing errors.
    for (size_t i = 0; i < neuron.output_connection_count; i++) {
        size_t outputConnectionIndex = neuron.output_connection_start_index + i;
        const OutputConnection& outputConnection = this->output_connections_[outputConnectionIndex];
        const InputConnection& inputConnection = this->input_connections_[outputConnection.input_connection_index];
        const Neuron& toNeuron = this->neurons_[inputConnection.to_neuron_index];

        sum += this->weights_[outputConnection.input_connection_index] * toNeuron.error;
    }

    neuron.error = ExecuteActivationFunctionDerivative(&neuron) * sum;
}

void NeuralNetwork::RunForward(const std::vector<double>* input) {
    assert(this->is_constructed_);
    assert(input->size() == this->input_neuron_count_);

    // Feed each input into the corresponding input neuron.
    size_t inputNeuronStartIndex = GetInputNeuronStartIndex();
    for (size_t i = 0; i < this->input_neuron_count_; i++) {
        this->neurons_[inputNeuronStartIndex + i].value = input->operator[](i);
    }

    // Pull the values from the input layer through the hidden layer neurons.
    size_t hiddenNeuronStartIndex = GetHiddenNeuronStartIndex();
    this->ComputeNeuronValueRange(hiddenNeuronStartIndex, this->hidden_neuron_count_);

    // Pull values into the output layer.
    size_t outputNeuronStartIndex = GetOutputNeuronStartIndex();
    this->ComputeNeuronValueRange(outputNeuronStartIndex, this->output_neuron_count_);
}

// static
double NeuralNetwork::ApplyErrorShaping(double value) {
    // TODO: Should this be replaced by?
    //return tanh(value);

    if (value < -0.9999999) {
        return -17.0;
    } else if (value > 0.9999999) {
        return 17.0;
    } else {
        return log((1.0 + value) / (1.0 - value));
    }
}

void NeuralNetwork::RunBackward(const std::vector<double>* output) {
    assert(this->is_constructed_);
    assert(output->size() == this->output_neuron_count_);

    this->ResetOutputLayerError();

    this->CalculateOutputLayerError(output);

    // Calculate error at each hidden layer neuron.
    size_t hiddenNeuronStartIndex = this->GetHiddenNeuronStartIndex();
    for (size_t i = 0; i < this->hidden_neuron_count_; i++) {
        ComputeNeuronError(hiddenNeuronStartIndex + (this->hidden_neuron_count_ - 1 - i));
    }
}

void NeuralNetwork::InitializeWeightsRandom(double min, double max) {
    assert(this->is_constructed_);

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
    const double factor = pow(neuron_percentage * hidden_neuron_count_, 1.0 / hidden_neuron_count_) / (max_input - min_input);
    InitializeWeightsRandom(-factor, factor);
}

// static
bool NeuralNetwork::IsActivationFunctionSymmetric(ActivationFunctionType activationFunctionType) {
    return activationFunctionType >= ActivationFunctionType::FirstSymmetric;
}

// static
double NeuralNetwork::ExecuteActivationFunction(Neuron* neuron) {
    switch (neuron->activation_function_type) {
    case ActivationFunctionType::Sigmoid:
        return ActivationFunction::ExecuteSigmoid(neuron->field);
    case ActivationFunctionType::SigmoidSymmetric:
        return ActivationFunction::ExecuteSigmoidSymmetric(neuron->field);
    case ActivationFunctionType::Gaussian:
        return ActivationFunction::ExecuteGaussian(neuron->field);
    case ActivationFunctionType::GaussianSymmetric:
        return ActivationFunction::ExecuteGaussianSymmetric(neuron->field);
    case ActivationFunctionType::Cosine:
        return ActivationFunction::ExecuteCosine(neuron->field);
    case ActivationFunctionType::CosineSymmetric:
        return ActivationFunction::ExecuteCosineSymmetric(neuron->field);
    case ActivationFunctionType::Sine:
        return ActivationFunction::ExecuteSine(neuron->field);
    case ActivationFunctionType::SineSymmetric:
        return ActivationFunction::ExecuteSineSymmetric(neuron->field);
    case ActivationFunctionType::Elliot:
        return ActivationFunction::ExecuteElliot(neuron->field);
    case ActivationFunctionType::ElliotSymmetric:
        return ActivationFunction::ExecuteElliotSymmetric(neuron->field);
    case ActivationFunctionType::Linear:
        return ActivationFunction::ExecuteLinear(neuron->field);
    case ActivationFunctionType::Threshold:
        return ActivationFunction::ExecuteThreshold(neuron->field);
    case ActivationFunctionType::ThresholdSymmetric:
        return ActivationFunction::ExecuteThresholdSymmetric(neuron->field);
    default:
        assert(false);
    }

    return 0;
}

// static
double NeuralNetwork::ExecuteActivationFunctionDerivative(Neuron* neuron) {
    switch (neuron->activation_function_type) {
    case ActivationFunctionType::Sigmoid:
        return ActivationFunction::ExecuteDerivativeSigmoid(neuron->value);
    case ActivationFunctionType::SigmoidSymmetric:
        return ActivationFunction::ExecuteDerivativeSigmoidSymmetric(neuron->value);
    case ActivationFunctionType::Gaussian:
        return ActivationFunction::ExecuteDerivativeGaussian(neuron->value, neuron->field);
    case ActivationFunctionType::GaussianSymmetric:
        return ActivationFunction::ExecuteDerivativeGaussianSymmetric(neuron->value, neuron->field);
    case ActivationFunctionType::Cosine:
        return ActivationFunction::ExecuteDerivativeCosine(neuron->field);
    case ActivationFunctionType::CosineSymmetric:
        return ActivationFunction::ExecuteDerivativeCosineSymmetric(neuron->field);
    case ActivationFunctionType::Sine:
        return ActivationFunction::ExecuteDerivativeSine(neuron->field);
    case ActivationFunctionType::SineSymmetric:
        return ActivationFunction::ExecuteDerivativeSineSymmetric(neuron->field);
    case ActivationFunctionType::Elliot:
        return ActivationFunction::ExecuteDerivativeElliot(neuron->field);
    case ActivationFunctionType::ElliotSymmetric:
        return ActivationFunction::ExecuteDerivativeElliotSymmetric(neuron->field);
    case ActivationFunctionType::Linear:
        return ActivationFunction::ExecuteDerivativeLinear(neuron->value);
    default:
        assert(false);
    }

    return 0;
}

double NeuralNetwork::GetError() const {
    assert(this->error_count_ != 0);

    return this->error_sum_ / this->error_count_;
}

double NeuralNetwork::GetError(const std::vector<double>* output) {
    this->ResetOutputLayerError();
    this->CalculateOutputLayerError(output);
    return this->GetError();
}

double NeuralNetwork::GetError(const TrainingData* trainingData) {
    this->ResetOutputLayerError();

    for (size_t i = 0; i < trainingData->size(); i++) {
        this->RunForward(&trainingData->operator[](i).input);
        this->CalculateOutputLayerError(&trainingData->operator[](i).output);
    }

    return this->GetError();
}

void NeuralNetwork::ResetOutputLayerError() {
    this->error_count_ = 0;
    this->error_sum_ = 0.0;
}

void NeuralNetwork::CalculateOutputLayerError(const std::vector<double>* output) {
    // Calculate error at each output neuron.
    size_t outputNeuronStartIndex = this->GetOutputNeuronStartIndex();
    for (size_t i = 0; i < this->output_neuron_count_; i++) {
        Neuron& neuron = this->neurons_[outputNeuronStartIndex + i];
        double delta = neuron.value - output->operator[](i);

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

        neuron.error = ExecuteActivationFunctionDerivative(&neuron) * delta;
    }
}

std::vector<double>& NeuralNetwork::GetWeights() {
    return this->weights_;
}

void NeuralNetwork::SetWeights(std::vector<double>& weights) {
    assert(weights_.size() == weights.size());

    weights_.assign(weights.cbegin(), weights.cend());
}

void NeuralNetwork::GetOutput(std::vector<double>* output) const {
    output->resize(output_neuron_count_);
    size_t firstOutputNeuron = GetOutputNeuronStartIndex();
    for (size_t i = 0; i < output_neuron_count_; i++) {
        output->operator[](i) = neurons_[firstOutputNeuron + i].value;
    }
}

}  // namespace panann
