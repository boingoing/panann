//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#include <cassert>
#include <algorithm>

#include "NeuralNetwork.h"
#include "ActivationFunction.h"
#include "TrainingData.h"

using namespace panann;

NeuralNetwork::NeuralNetwork() :
    _inputNeuronCount(0),
    _outputNeuronCount(0),
    _hiddenNeuronCount(0),
    _learningRate(0.7),
    _momentum(0.1),
    _qpropMu(1.75),
    _qpropWeightDecay(-0.0001),
    _rpropWeightStepInitial(0.0125),
    _rpropWeightStepMin(0.000001),
    _rpropWeightStepMax(50.0),
    _rpropIncreaseFactor(1.2),
    _rpropDecreaseFactor(0.5),
    _sarpropWeightDecayShift(0.01),
    _sarpropStepThresholdFactor(0.1),
    _sarpropStepShift(3),
    _sarpropTemperature(0.015),
    _errorSum(0.0),
    _errorCount(0),
    _hiddenNeuronActivationFunctionType(ActivationFunctionType::Sigmoid),
    _outputNeuronActivationFunctionType(ActivationFunctionType::Sigmoid),
    _errorCostFunction(ErrorCostFunction::MeanSquareError),
    _trainingAlgorithmType(TrainingAlgorithmType::ResilientBackpropagation),
    _shouldShapeErrorCurve(true),
    _enableShortcutConnections(false),
    _isConstructed(false) {
}

size_t NeuralNetwork::GetInputNeuronCount() {
    return this->_inputNeuronCount;
}

void NeuralNetwork::SetInputNeuronCount(size_t inputNeuronCount) {
    assert(!this->_isConstructed);
    this->_inputNeuronCount = inputNeuronCount;
}

size_t NeuralNetwork::GetOutputNeuronCount() {
    return this->_outputNeuronCount;
}

void NeuralNetwork::SetOutputNeuronCount(size_t outputNeuronCount) {
    assert(!this->_isConstructed);
    this->_outputNeuronCount = outputNeuronCount;
}

void NeuralNetwork::SetLearningRate(double learningRate) {
    this->_learningRate = learningRate;
}

double NeuralNetwork::GetLearningRate() {
    return this->_learningRate;
}

void NeuralNetwork::SetMomentum(double momentum) {
    this->_momentum = momentum;
}

double NeuralNetwork::GetMomentum() {
    return this->_momentum;
}

void NeuralNetwork::SetQpropMu(double mu) {
    this->_qpropMu = mu;
}

double NeuralNetwork::GetQpropMu() {
    return this->_qpropMu;
}

void NeuralNetwork::SetQpropWeightDecay(double weightDecay) {
    this->_qpropWeightDecay = weightDecay;
}

double NeuralNetwork::GetQpropWeightDecay() {
    return this->_qpropWeightDecay;
}

void NeuralNetwork::SetRpropWeightStepInitial(double weightStep) {
    this->_rpropWeightStepInitial = weightStep;
}

double NeuralNetwork::GetRpropWeightStepInitial() {
    return this->_rpropWeightStepInitial;
}

void NeuralNetwork::SetRpropWeightStepMin(double weightStep) {
    this->_rpropWeightStepMin = weightStep;
}

double NeuralNetwork::GetRpropWeightStepMin() {
    return this->_rpropWeightStepMin;
}

void NeuralNetwork::SetRpropWeightStepMax(double weightStep) {
    this->_rpropWeightStepMax = weightStep;
}

double NeuralNetwork::GetRpropWeightStepMax() {
    return this->_rpropWeightStepMax;
}

void NeuralNetwork::SetRpropIncreaseFactor(double factor) {
    this->_rpropIncreaseFactor = factor;
}

double NeuralNetwork::GetRpropIncreaseFactor() {
    return this->_rpropIncreaseFactor;
}

void NeuralNetwork::SetRpropDecreaseFactor(double factor) {
    this->_rpropDecreaseFactor = factor;
}

double NeuralNetwork::GetRpropDecreaseFactor() {
    return this->_rpropDecreaseFactor;
}

void NeuralNetwork::SetSarpropWeightDecayShift(double k1) {
    this->_sarpropWeightDecayShift = k1;
}

double NeuralNetwork::GetSarpropWeightDecayShift() {
    return this->_sarpropWeightDecayShift;
}

void NeuralNetwork::SetSarpropStepThresholdFactor(double k2) {
    this->_sarpropStepThresholdFactor = k2;
}

double NeuralNetwork::GetSarpropStepThresholdFactor() {
    return this->_sarpropStepThresholdFactor;
}

void NeuralNetwork::SetSarpropStepShift(double k3) {
    this->_sarpropStepShift = k3;
}

double NeuralNetwork::GetSarpropStepShift() {
    return this->_sarpropStepShift;
}

void NeuralNetwork::SetSarpropTemperature(double t) {
    this->_sarpropTemperature = t;
}

double NeuralNetwork::GetSarpropTemperature() {
    return this->_sarpropTemperature;
}

void NeuralNetwork::SetTrainingAlgorithmType(TrainingAlgorithmType type) {
    this->_trainingAlgorithmType = type;
}

NeuralNetwork::TrainingAlgorithmType NeuralNetwork::GetTrainingAlgorithmType() {
    return this->_trainingAlgorithmType;
}

void NeuralNetwork::EnableShortcutConnections() {
    assert(!this->_isConstructed);
    this->_enableShortcutConnections = true;
}

void NeuralNetwork::DisableShortcutConnections() {
    assert(!this->_isConstructed);
    this->_enableShortcutConnections = false;
}

void NeuralNetwork::SetHiddenNeuronActivationFunctionType(ActivationFunctionType type) {
    assert(!this->_isConstructed);
    this->_hiddenNeuronActivationFunctionType = type;
}

NeuralNetwork::ActivationFunctionType NeuralNetwork::GetHiddenNeuronActivationFunctionType() {
    return this->_hiddenNeuronActivationFunctionType;
}

void NeuralNetwork::SetOutputNeuronActivationFunctionType(ActivationFunctionType type) {
    assert(!this->_isConstructed);
    this->_outputNeuronActivationFunctionType = type;
}

NeuralNetwork::ActivationFunctionType NeuralNetwork::GetOutputNeuronActivationFunctionType() {
    return this->_outputNeuronActivationFunctionType;
}

void NeuralNetwork::SetNeuronActivationFunction(size_t neuronIndex, ActivationFunctionType type) {
    assert(this->_isConstructed);
    assert(neuronIndex < this->_neurons.size());
    this->_neurons[neuronIndex]._activationFunctionType = type;
}

void NeuralNetwork::SetErrorCostFunction(ErrorCostFunction mode) {
    this->_errorCostFunction = mode;
}

NeuralNetwork::ErrorCostFunction NeuralNetwork::GetErrorCostFunction() {
    return this->_errorCostFunction;
}

/**
 * | hidden neurons | input neurons | output neurons | bias neurons |
 */
size_t NeuralNetwork::GetInputNeuronStartIndex() {
    return this->_hiddenNeuronCount;
}

size_t NeuralNetwork::GetOutputNeuronStartIndex() {
    return this->_hiddenNeuronCount + this->_inputNeuronCount;
}

size_t NeuralNetwork::GetHiddenNeuronStartIndex() {
    return 0;
}

size_t NeuralNetwork::GetHiddenNeuronCount() {
    return this->_hiddenNeuronCount;
}

size_t NeuralNetwork::GetBiasNeuronStartIndex() {
    return this->_inputNeuronCount + this->_outputNeuronCount + this->_hiddenNeuronCount;
}

void NeuralNetwork::AddHiddenLayer(size_t neuronCount) {
    assert(!this->_isConstructed);

    Layer layer;
    layer._neuronStartIndex = this->GetHiddenNeuronStartIndex() + this->_hiddenNeuronCount;
    layer._neuronCount = neuronCount;

    this->_hiddenLayers.push_back(layer);
    this->_hiddenNeuronCount += neuronCount;
}

void NeuralNetwork::Allocate() {
    assert(!this->_isConstructed);
    // Do not support networks with no hidden layers.
    assert(!this->_hiddenLayers.empty());

    // Total count of neurons is all the input, output, and hidden neurons.
    // The input layer and each hidden layer also contribute one bias neuron.
    size_t neuronCount =
        this->_inputNeuronCount + 1 +
        this->_outputNeuronCount +
        this->_hiddenNeuronCount + this->_hiddenLayers.size();

    this->_neurons.resize(neuronCount);

    size_t biasNeuronIndex = this->GetBiasNeuronStartIndex();
    size_t inputConnectionIndex = 0;
    size_t outputConnectionIndex = 0;
    size_t currentLayerInputConnectionCount = 0;
    size_t currentLayerOutputConnectionCount = 0;

    // Calculate the connections outgoing from the input layer.
    if (this->_enableShortcutConnections) {
        // The input layer connects to all hidden layers and the output layer.
        currentLayerOutputConnectionCount = this->_hiddenNeuronCount + this->_outputNeuronCount;
    } else {
        // The input layer connects only to the first hidden layer.
        currentLayerOutputConnectionCount = this->_hiddenLayers.front()._neuronCount;
    }

    size_t inputNeuronIndex = this->GetInputNeuronStartIndex();
    for (size_t i = 0; i < this->_inputNeuronCount; i++) {
        Neuron& neuron = this->_neurons[inputNeuronIndex + i];
        neuron._outputConnectionStartIndex = outputConnectionIndex;
        outputConnectionIndex += currentLayerOutputConnectionCount;
    }

    // The first bias neuron is the one for the input layer.
    Neuron& biasNeuronInput = this->_neurons[biasNeuronIndex++];
    biasNeuronInput._outputConnectionStartIndex = outputConnectionIndex;
    biasNeuronInput._value = 1.0;
    outputConnectionIndex += this->_hiddenLayers.front()._neuronCount;

    // Calculate the connections incoming to the output layer.
    if (this->_enableShortcutConnections) {
        // All input and hidden neurons are connected to each output neuron.
        currentLayerInputConnectionCount = this->_inputNeuronCount + this->_hiddenNeuronCount + 1;
    } else {
        // Output neurons are connected only to the last hidden layer.
        currentLayerInputConnectionCount = this->_hiddenLayers.back()._neuronCount + 1;
    }

    size_t firstOutputNeuronIndex = this->GetOutputNeuronStartIndex();
    for (size_t i = 0; i < this->_outputNeuronCount; i++) {
        Neuron& neuron = this->_neurons[firstOutputNeuronIndex + i];
        neuron._inputConnectionStartIndex = inputConnectionIndex;
        neuron._activationFunctionType = this->_outputNeuronActivationFunctionType;
        inputConnectionIndex += currentLayerInputConnectionCount;
    }

    size_t neuronIndex = this->GetHiddenNeuronStartIndex();

    // Calculate the connections to and from all hidden layers.
    for (size_t layerIndex = 0; layerIndex < this->_hiddenLayers.size(); layerIndex++) {
        currentLayerInputConnectionCount = 0;
        currentLayerOutputConnectionCount = 0;

        if (this->_enableShortcutConnections) {
            // All hidden layers connect to the input layer when shortcuts are enabled.
            currentLayerInputConnectionCount += this->_inputNeuronCount + 1;

            // Each neuron in this layer connects to the neurons in all previous hidden layers.
            for (size_t previousLayerIndex = 0; previousLayerIndex < layerIndex; previousLayerIndex++) {
                currentLayerInputConnectionCount += this->_hiddenLayers[previousLayerIndex]._neuronCount + 1;
            }

            // All hidden layers connect directly to the output layer when shortcuts are enabled.
            currentLayerOutputConnectionCount += this->_outputNeuronCount;

            // This layer connects to all neurons in subsequent hidden layers.
            for (size_t nextLayerIndex = layerIndex + 1; nextLayerIndex < this->_hiddenLayers.size(); nextLayerIndex++) {
                currentLayerOutputConnectionCount += this->_hiddenLayers[nextLayerIndex]._neuronCount;
            }
        } else {
            if (layerIndex == 0) {
                // First hidden layer connects to the input layer.
                currentLayerInputConnectionCount += this->_inputNeuronCount + 1;
            } else {
                // This hidden layer connects directly the previous one.
                currentLayerInputConnectionCount += this->_hiddenLayers[layerIndex - 1]._neuronCount + 1;
            }

            if (layerIndex == this->_hiddenLayers.size() - 1) {
                // Last hidden layer connects to the output layer.
                currentLayerOutputConnectionCount += this->_outputNeuronCount;
            } else {
                assert(layerIndex + 1 < this->_hiddenLayers.size());

                // This hidden layer connects directly to the next one.
                currentLayerOutputConnectionCount += this->_hiddenLayers[layerIndex + 1]._neuronCount;
            }
        }

        const Layer& currentLayer = this->_hiddenLayers[layerIndex];
        for (size_t i = 0; i < currentLayer._neuronCount; i++) {
            Neuron& neuron = this->_neurons[neuronIndex++];
            neuron._inputConnectionStartIndex = inputConnectionIndex;
            neuron._outputConnectionStartIndex = outputConnectionIndex;
            neuron._activationFunctionType = this->_hiddenNeuronActivationFunctionType;

            inputConnectionIndex += currentLayerInputConnectionCount;
            outputConnectionIndex += currentLayerOutputConnectionCount;
        }

        // Bias neurons cannot have shortcut connections.
        size_t biasOutputConnections = 0;
        if (layerIndex == this->_hiddenLayers.size() - 1) {
            // Bias neuron in the last hidden layer connects to the output layer.
            biasOutputConnections = this->_outputNeuronCount;
        } else {
            // Bias neuron in this hidden layer connects to the next hidden layer.
            biasOutputConnections = this->_hiddenLayers[layerIndex + 1]._neuronCount;
        }

        // Bias neurons do not have incoming connections.
        Neuron& biasNeuron = this->_neurons[biasNeuronIndex++];
        biasNeuron._outputConnectionStartIndex = outputConnectionIndex;
        biasNeuron._value = 1.0;
        outputConnectionIndex += biasOutputConnections;
    }

    this->_inputConnections.resize(inputConnectionIndex);
    this->_outputConnections.resize(outputConnectionIndex);
    this->_weights.resize(inputConnectionIndex);
}

void NeuralNetwork::ConnectNeurons(size_t fromNeuronIndex, size_t toNeuronIndex) {
    Neuron& fromNeuron = this->_neurons[fromNeuronIndex];
    Neuron& toNeuron = this->_neurons[toNeuronIndex];

    size_t inputConnectionIndex = toNeuron._inputConnectionStartIndex + toNeuron._inputConnectionCount;
    InputConnection& inputConnection = this->_inputConnections.at(inputConnectionIndex);
    inputConnection._fromNeuronIndex = fromNeuronIndex;
    inputConnection._toNeuronIndex = toNeuronIndex;
    toNeuron._inputConnectionCount++;

    size_t outputConnectionIndex = fromNeuron._outputConnectionStartIndex + fromNeuron._outputConnectionCount;
    OutputConnection& outputConnection = this->_outputConnections.at(outputConnectionIndex);
    outputConnection._inputConnectionIndex = inputConnectionIndex;
    fromNeuron._outputConnectionCount++;
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

void NeuralNetwork::ConnectBiasNeuron(size_t biasNeuronIndex, size_t toNeuronIndex, size_t toNeuronCount) {
    ConnectLayers(biasNeuronIndex, 1, toNeuronIndex, toNeuronCount);
}

void NeuralNetwork::ConnectFully() {
    assert(!this->_isConstructed);
    assert(!this->_hiddenLayers.empty());

    size_t inputNeuronStartIndex = this->GetInputNeuronStartIndex();
    size_t biasNeuronIndex = this->GetBiasNeuronStartIndex();

    for (size_t layerIndex = 0; layerIndex < this->_hiddenLayers.size(); layerIndex++) {
        const Layer& currentLayer = this->_hiddenLayers[layerIndex];

        if (this->_enableShortcutConnections) {
            // Connect to input layer.
            ConnectLayers(inputNeuronStartIndex,
                          this->_inputNeuronCount,
                          currentLayer._neuronStartIndex,
                          currentLayer._neuronCount);

            // Connect to all previous hidden layers.
            for (size_t previousLayerIndex = 0; previousLayerIndex < layerIndex; previousLayerIndex++) {
                const Layer& previousLayer = this->_hiddenLayers[previousLayerIndex];
                ConnectLayers(previousLayer._neuronStartIndex,
                              previousLayer._neuronCount,
                              currentLayer._neuronStartIndex,
                              currentLayer._neuronCount);
            }
        } else {
            if (layerIndex == 0) {
                // Connect first hidden layer to input layer.
                ConnectLayers(inputNeuronStartIndex,
                              this->_inputNeuronCount,
                              currentLayer._neuronStartIndex,
                              currentLayer._neuronCount);
            } else {
                // Connect to previous hidden layer.
                const Layer& previousLayer = this->_hiddenLayers[layerIndex - 1];
                ConnectLayers(previousLayer._neuronStartIndex,
                              previousLayer._neuronCount,
                              currentLayer._neuronStartIndex,
                              currentLayer._neuronCount);
            }
        }

        // Bias neurons do not have shortcut connections.
        // Just connect this layer to the bias neuron in the layer before it.
        ConnectBiasNeuron(biasNeuronIndex++, currentLayer._neuronStartIndex, currentLayer._neuronCount);
    }

    size_t outputNeuronStartIndex = this->GetOutputNeuronStartIndex();
    if (this->_enableShortcutConnections) {
        // Connect input layer to output layer.
        ConnectLayers(inputNeuronStartIndex,
                      this->_inputNeuronCount,
                      outputNeuronStartIndex,
                      this->_outputNeuronCount);

        // Connect all hidden layers to output layer.
        for (size_t i = 0; i < this->_hiddenLayers.size(); i++) {
            const Layer& layer = this->_hiddenLayers[i];
            ConnectLayers(layer._neuronStartIndex,
                          layer._neuronCount,
                          outputNeuronStartIndex,
                          this->_outputNeuronCount);
        }
    } else {
        const Layer& previousLayer = this->_hiddenLayers.back();
        // Connect output layer to the last hidden layer.
        ConnectLayers(previousLayer._neuronStartIndex,
                      previousLayer._neuronCount,
                      outputNeuronStartIndex,
                      this->_outputNeuronCount);
    }

    // Connect output layer to the bias neuron in the last hidden layer.
    ConnectBiasNeuron(biasNeuronIndex, outputNeuronStartIndex, this->_outputNeuronCount);
}

void NeuralNetwork::Construct() {
    assert(!this->_isConstructed);

    this->Allocate();
    this->ConnectFully();

    this->_isConstructed = true;
}

void NeuralNetwork::ResetWeightSteps() {
    double initialWeightStep =
        (this->_trainingAlgorithmType == TrainingAlgorithmType::ResilientBackpropagation ||
         this->_trainingAlgorithmType == TrainingAlgorithmType::SimulatedAnnealingResilientBackpropagation) ?
        this->_rpropWeightStepInitial :
        0;

    this->_previousWeightSteps.resize(this->_weights.size());
    std::fill(this->_previousWeightSteps.begin(), this->_previousWeightSteps.end(), initialWeightStep);
}

void NeuralNetwork::ResetSlopes() {
    this->_slopes.resize(this->_weights.size());
    std::fill(this->_slopes.begin(), this->_slopes.end(), 0);
}

void NeuralNetwork::ResetPreviousSlopes() {
    this->_previousSlopes.resize(this->_weights.size());
    std::fill(this->_previousSlopes.begin(), this->_previousSlopes.end(), 0);
}

void NeuralNetwork::UpdateWeightsOnline() {
    for (size_t i = 0; i < this->_inputConnections.size(); i++) {
        const InputConnection& connection = this->_inputConnections[i];
        const Neuron& fromNeuron = this->_neurons[connection._fromNeuronIndex];
        const Neuron& toNeuron = this->_neurons[connection._toNeuronIndex];

        double delta = -1.0 * this->_learningRate * toNeuron._error * fromNeuron._value + this->_momentum * this->_previousWeightSteps[i];
        this->_previousWeightSteps[i] = delta;
        this->_weights[i] += delta;
    }
}

void NeuralNetwork::UpdateWeightsOffline(size_t currentEpoch, size_t stepCount) {
    assert(stepCount != 0);

    switch (this->_trainingAlgorithmType) {
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
    double epsilon = this->_learningRate / stepCount;

    for (size_t i = 0; i < this->_weights.size(); i++) {
        this->_weights[i] += this->_slopes[i] * epsilon;
    }
}

void NeuralNetwork::UpdateWeightsQuickBackpropagation(size_t stepCount) {
    double epsilon = this->_learningRate / stepCount;
    double shrinkFactor = this->_qpropMu / (1.0 + this->_qpropMu);

    for (size_t i = 0; i < this->_weights.size(); i++) {
        double previousSlope = this->_previousSlopes[i];
        double previousWeightStep = this->_previousWeightSteps[i];
        double currentSlope = this->_slopes[i] + this->_qpropWeightDecay * this->_weights[i];
        double weightStep = epsilon * currentSlope;

        if (previousWeightStep > 0.001) {
            if (currentSlope <= 0.0) {
                weightStep = 0.0;
            }

            if (currentSlope > (shrinkFactor * previousSlope)) {
                weightStep += this->_qpropMu * previousWeightStep;
            } else {
                weightStep += previousWeightStep * currentSlope / (previousSlope - currentSlope);
            }
        } else if (previousWeightStep < -0.001) {
            if (currentSlope >= 0.0) {
                weightStep = 0.0;
            }

            if (currentSlope < (shrinkFactor * previousSlope)) {
                weightStep += this->_qpropMu * previousWeightStep;
            } else {
                weightStep += previousWeightStep * currentSlope / (previousSlope - currentSlope);
            }
        }

        this->_previousSlopes[i] = currentSlope;
        this->_previousWeightSteps[i] = weightStep;

        this->_weights[i] += weightStep;
    }
}

void NeuralNetwork::UpdateWeightsResilientBackpropagation() {
    for (size_t i = 0; i < this->_weights.size(); i++) {
        double previousSlope = this->_previousSlopes[i];
        double currentSlope = this->_slopes[i];
        double weightStep = 0.0;
        double previousSlopeTimesCurrentSlope = previousSlope * currentSlope;
        double previousWeightStep = std::max(this->_previousWeightSteps[i], this->_rpropWeightStepMin);

        if (previousSlopeTimesCurrentSlope >= 0.0) {
            weightStep = std::min(previousWeightStep * this->_rpropIncreaseFactor, this->_rpropWeightStepMax);
        } else if (previousSlopeTimesCurrentSlope < 0.0) {
            weightStep = std::max(previousWeightStep * this->_rpropDecreaseFactor, this->_rpropWeightStepMin);
            currentSlope = 0.0;
        }

        double weightDelta = std::signbit(currentSlope) ? -1 * weightStep : weightStep;
        this->_weights[i] += weightDelta;

        this->_previousSlopes[i] = currentSlope;
        this->_previousWeightSteps[i] = weightStep;
    }
}

void NeuralNetwork::UpdateWeightsSimulatedAnnealingResilientBackpropagation(size_t currentEpoch) {
    for (size_t i = 0; i < this->_weights.size(); i++) {
        double previousSlope = this->_previousSlopes[i];
        double currentSlope = this->_slopes[i] - this->_sarpropWeightDecayShift * this->_weights[i] * std::exp2(-this->_sarpropTemperature * currentEpoch);
        double previousSlopeTimesCurrentSlope = previousSlope * currentSlope;
        double previousWeightStep = std::max(this->_previousWeightSteps[i], this->_rpropWeightStepMin);
        double weightStep = 0.0;

        if (previousSlopeTimesCurrentSlope > 0.0) {
            weightStep = std::min(previousWeightStep * this->_rpropIncreaseFactor, this->_rpropWeightStepMax);
            double weightDelta = std::signbit(currentSlope) ? -1 * weightStep : weightStep;
            this->_weights[i] += weightDelta;
        } else if (previousSlopeTimesCurrentSlope < 0.0) {
            double rmsError = std::sqrt(this->GetError());

            if (previousWeightStep < this->_sarpropStepThresholdFactor * rmsError * rmsError) {
                weightStep = previousWeightStep * this->_rpropDecreaseFactor + this->_sarpropStepShift * this->_randomWrapper.RandomFloat(0.0, 1.0) * rmsError * std::exp2(-this->_sarpropTemperature * currentEpoch);
            } else {
                weightStep = std::max(previousWeightStep * this->_rpropDecreaseFactor, this->_rpropWeightStepMin);
            }

            currentSlope = 0.0;
        } else {
            double weightDelta = std::signbit(currentSlope) ? -1 * previousWeightStep : previousWeightStep;
            this->_weights[i] += weightDelta;
        }

        this->_previousSlopes[i] = currentSlope;
        this->_previousWeightSteps[i] = weightStep;
    }
}

void NeuralNetwork::UpdateSlopes() {
    for (size_t i = 0; i < this->_inputConnections.size(); i++) {
        const InputConnection& connection = this->_inputConnections[i];
        const Neuron& fromNeuron = this->_neurons[connection._fromNeuronIndex];
        const Neuron& toNeuron = this->_neurons[connection._toNeuronIndex];

        this->_slopes[i] += -1.0 * fromNeuron._value * toNeuron._error;
    }
}

void NeuralNetwork::TrainOffline(TrainingData* trainingData, size_t epochCount) {
    this->ResetPreviousSlopes();
    this->ResetWeightSteps();

    for (size_t i = 0; i < epochCount; i++) {
        this->_randomWrapper.ShuffleVector(trainingData);
        this->ResetSlopes();

        // Train the network using offline weight updates - batching
        for (size_t j = 0; j < trainingData->size(); j++) {
            // Run the network forward to get values in the output neurons.
            this->RunForward(&trainingData->operator[](j)._input);

            // Run the network backward to propagate the error values
            this->RunBackward(&trainingData->operator[](j)._output);

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
        this->_randomWrapper.ShuffleVector(trainingData);

        // Train the network using online weight updates - no batching
        for (size_t j = 0; j < trainingData->size(); j++) {
            // Run the network forward to get values in the output neurons.
            this->RunForward(&trainingData->operator[](j)._input);

            // Run the network backward to propagate the error values
            this->RunBackward(&trainingData->operator[](j)._output);

            // Update weights online - no batching
            this->UpdateWeightsOnline();
        }
    }
}

void NeuralNetwork::Train(TrainingData* trainingData, size_t epochCount) {
    switch (this->_trainingAlgorithmType) {
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

void NeuralNetwork::ComputeNeuronValue(size_t neuronIndex) {
    Neuron& neuron = this->_neurons[neuronIndex];
    neuron._field = 0.0;

    // Sum incoming values.
    for (size_t i = 0; i < neuron._inputConnectionCount; i++) {
        size_t inputConnectionIndex = neuron._inputConnectionStartIndex + i;
        const InputConnection& connection = this->_inputConnections[inputConnectionIndex];
        const Neuron& fromNeuron = this->_neurons[connection._fromNeuronIndex];

        neuron._field += fromNeuron._value * this->_weights[inputConnectionIndex];
    }

    neuron._value = ExecuteActivationFunction(&neuron);
}

void NeuralNetwork::ComputeNeuronError(size_t neuronIndex) {
    Neuron& neuron = this->_neurons[neuronIndex];
    double sum = 0.0;

    // Sum outgoing errors.
    for (size_t i = 0; i < neuron._outputConnectionCount; i++) {
        size_t outputConnectionIndex = neuron._outputConnectionStartIndex + i;
        const OutputConnection& outputConnection = this->_outputConnections[outputConnectionIndex];
        const InputConnection& inputConnection = this->_inputConnections[outputConnection._inputConnectionIndex];
        const Neuron& toNeuron = this->_neurons[inputConnection._toNeuronIndex];

        sum += this->_weights[outputConnection._inputConnectionIndex] * toNeuron._error;
    }

    neuron._error = ExecuteActivationFunctionDerivative(&neuron) * sum;
}

void NeuralNetwork::RunForward(const std::vector<double>* input) {
    assert(this->_isConstructed);
    assert(input->size() == this->_inputNeuronCount);

    // Feed each input into the corresponding input neuron.
    size_t inputNeuronStartIndex = GetInputNeuronStartIndex();
    for (size_t i = 0; i < this->_inputNeuronCount; i++) {
        this->_neurons[inputNeuronStartIndex + i]._value = input->operator[](i);
    }

    // Pull the values from the input layer through the hidden layer neurons.
    size_t hiddenNeuronStartIndex = GetHiddenNeuronStartIndex();
    for (size_t i = 0; i < this->_hiddenNeuronCount; i++) {
        ComputeNeuronValue(hiddenNeuronStartIndex + i);
    }

    // Pull values into the output layer.
    size_t outputNeuronStartIndex = GetOutputNeuronStartIndex();
    for (size_t i = 0; i < this->_outputNeuronCount; i++) {
        ComputeNeuronValue(outputNeuronStartIndex + i);
    }
}

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
    assert(this->_isConstructed);
    assert(output->size() == this->_outputNeuronCount);

    this->ResetOutputLayerError();

    this->CalculateOutputLayerError(output);

    // Calculate error at each hidden layer neuron.
    size_t hiddenNeuronStartIndex = this->GetHiddenNeuronStartIndex();
    for (size_t i = 0; i < this->_hiddenNeuronCount; i++) {
        ComputeNeuronError(hiddenNeuronStartIndex + (this->_hiddenNeuronCount - 1 - i));
    }
}

void NeuralNetwork::InitializeWeightsRandom(double min, double max) {
    assert(this->_isConstructed);

    for (size_t i = 0; i < this->_weights.size(); i++) {
        this->_weights[i] = this->_randomWrapper.RandomFloat(min, max);
    }
}

void NeuralNetwork::InitializeWeights(const TrainingData* trainingData) {
    assert(this->_isConstructed);
    assert(!trainingData->empty());

    double minInput = std::numeric_limits<double>::max();
    double maxInput = std::numeric_limits<double>::min();

    std::for_each(trainingData->begin(), trainingData->end(), [&](const Example& ex) {
        assert(this->GetInputNeuronCount() == ex._input.size());
        auto minmax = std::minmax_element(ex._input.begin(), ex._input.end());
        minInput = std::min(minInput, *minmax.first);
        maxInput = std::max(maxInput, *minmax.second);
    });

    double factor = pow(0.7 * this->_hiddenNeuronCount, 1.0 / this->_hiddenNeuronCount) / (maxInput - minInput);

    this->InitializeWeightsRandom(-factor, factor);
}

bool NeuralNetwork::IsActivationFunctionSymmetric(ActivationFunctionType activationFunctionType) {
    return activationFunctionType >= ActivationFunctionType::FirstSymmetric;
}

double NeuralNetwork::ExecuteActivationFunction(Neuron* neuron) {
    switch (neuron->_activationFunctionType) {
    case ActivationFunctionType::Sigmoid:
        return ActivationFunction::ExecuteSigmoid(neuron->_field);
    case ActivationFunctionType::SigmoidSymmetric:
        return ActivationFunction::ExecuteSigmoidSymmetric(neuron->_field);
    case ActivationFunctionType::Gaussian:
        return ActivationFunction::ExecuteGaussian(neuron->_field);
    case ActivationFunctionType::GaussianSymmetric:
        return ActivationFunction::ExecuteGaussianSymmetric(neuron->_field);
    case ActivationFunctionType::Cosine:
        return ActivationFunction::ExecuteCosine(neuron->_field);
    case ActivationFunctionType::CosineSymmetric:
        return ActivationFunction::ExecuteCosineSymmetric(neuron->_field);
    case ActivationFunctionType::Sine:
        return ActivationFunction::ExecuteSine(neuron->_field);
    case ActivationFunctionType::SineSymmetric:
        return ActivationFunction::ExecuteSineSymmetric(neuron->_field);
    case ActivationFunctionType::Elliot:
        return ActivationFunction::ExecuteElliot(neuron->_field);
    case ActivationFunctionType::ElliotSymmetric:
        return ActivationFunction::ExecuteElliotSymmetric(neuron->_field);
    case ActivationFunctionType::Linear:
        return ActivationFunction::ExecuteLinear(neuron->_field);
    case ActivationFunctionType::Threshold:
        return ActivationFunction::ExecuteThreshold(neuron->_field);
    case ActivationFunctionType::ThresholdSymmetric:
        return ActivationFunction::ExecuteThresholdSymmetric(neuron->_field);
    default:
        assert(false);
    }

    return 0;
}

double NeuralNetwork::ExecuteActivationFunctionDerivative(Neuron* neuron) {
    switch (neuron->_activationFunctionType) {
    case ActivationFunctionType::Sigmoid:
        return ActivationFunction::ExecuteDerivativeSigmoid(neuron->_value);
    case ActivationFunctionType::SigmoidSymmetric:
        return ActivationFunction::ExecuteDerivativeSigmoidSymmetric(neuron->_value);
    case ActivationFunctionType::Gaussian:
        return ActivationFunction::ExecuteDerivativeGaussian(neuron->_value, neuron->_field);
    case ActivationFunctionType::GaussianSymmetric:
        return ActivationFunction::ExecuteDerivativeGaussianSymmetric(neuron->_value, neuron->_field);
    case ActivationFunctionType::Cosine:
        return ActivationFunction::ExecuteDerivativeCosine(neuron->_field);
    case ActivationFunctionType::CosineSymmetric:
        return ActivationFunction::ExecuteDerivativeCosineSymmetric(neuron->_field);
    case ActivationFunctionType::Sine:
        return ActivationFunction::ExecuteDerivativeSine(neuron->_field);
    case ActivationFunctionType::SineSymmetric:
        return ActivationFunction::ExecuteDerivativeSineSymmetric(neuron->_field);
    case ActivationFunctionType::Elliot:
        return ActivationFunction::ExecuteDerivativeElliot(neuron->_field);
    case ActivationFunctionType::ElliotSymmetric:
        return ActivationFunction::ExecuteDerivativeElliotSymmetric(neuron->_field);
    case ActivationFunctionType::Linear:
        return ActivationFunction::ExecuteDerivativeLinear(neuron->_value);
    default:
        assert(false);
    }

    return 0;
}

double NeuralNetwork::GetError() {
    assert(this->_errorCount != 0);

    return this->_errorSum / this->_errorCount;
}

double NeuralNetwork::GetError(const std::vector<double>* output) {
    this->ResetOutputLayerError();
    this->CalculateOutputLayerError(output);
    return this->GetError();
}

double NeuralNetwork::GetError(const TrainingData* trainingData) {
    this->ResetOutputLayerError();

    for (size_t i = 0; i < trainingData->size(); i++) {
        this->RunForward(&trainingData->operator[](i)._input);
        this->CalculateOutputLayerError(&trainingData->operator[](i)._output);
    }

    return this->GetError();
}

void NeuralNetwork::ResetOutputLayerError() {
    this->_errorCount = 0;
    this->_errorSum = 0.0;
}

void NeuralNetwork::CalculateOutputLayerError(const std::vector<double>* output) {
    // Calculate error at each output neuron.
    size_t outputNeuronStartIndex = this->GetOutputNeuronStartIndex();
    for (size_t i = 0; i < this->_outputNeuronCount; i++) {
        Neuron& neuron = this->_neurons[outputNeuronStartIndex + i];
        double delta = neuron._value - output->operator[](i);

        switch (this->_errorCostFunction) {
        case ErrorCostFunction::MeanSquareError:
            this->_errorSum += (delta * delta) / 2;
            break;
        case ErrorCostFunction::MeanAbsoluteError:
            this->_errorSum += std::fabs(delta);
            break;
        }
        this->_errorCount++;

        /*
        if (IsActivationFunctionSymmetric(neuron._activationFunctionType)) {
            delta /= 2;
        }

        if (this->_shouldShapeErrorCurve) {
            delta = ApplyErrorShaping(delta);
        }
        */

        neuron._error = ExecuteActivationFunctionDerivative(&neuron) * delta;
    }
}

std::vector<double>* NeuralNetwork::GetWeights() {
    return &this->_weights;
}

void NeuralNetwork::SetWeights(std::vector<double>* weights) {
    assert(this->_weights.size() == weights->size());

    this->_weights.assign(weights->begin(), weights->end());
}

void NeuralNetwork::GetOutput(std::vector<double>* output) {
    output->resize(this->_outputNeuronCount);
    size_t firstOutputNeuron = this->GetOutputNeuronStartIndex();
    for (size_t i = 0; i < this->_outputNeuronCount; i++) {
        output->operator[](i) = this->_neurons[firstOutputNeuron + i]._value;
    }
}
