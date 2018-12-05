//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panga contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#include <cassert>

#include "NeuralNetwork.h"
#include "TrainingData.h"
#include "ActivationFunction.h"

NeuralNetwork::NeuralNetwork(size_t inputNeuronCount, size_t outputNeuronCount) :
    _inputNeuronCount(inputNeuronCount),
    _outputNeuronCount(outputNeuronCount),
    _hiddenNeuronCount(0),
    _shouldShapeErrorCurve(true),
    _enableShortcutConnections(true) {
}

size_t NeuralNetwork::GetInputNeuronCount() {
    return this->_inputNeuronCount;
}

size_t NeuralNetwork::GetOutputNeuronCount() {
    return this->_outputNeuronCount;
}

size_t NeuralNetwork::GetInputNeuronStartIndex() {
    return 0;
}

size_t NeuralNetwork::GetOutputNeuronStartIndex() {
    return this->_inputNeuronCount;
}

size_t NeuralNetwork::GetHiddenNeuronStartIndex() {
    return this->_inputNeuronCount + this->_outputNeuronCount;
}

size_t NeuralNetwork::GetBiasNeuronStartIndex() {
    return this->_inputNeuronCount + this->_outputNeuronCount + this->_hiddenNeuronCount;
}

void NeuralNetwork::AddHiddenLayer(size_t neuronCount) {
    Layer layer;
    layer._neuronStartIndex = this->GetHiddenNeuronStartIndex() + this->_hiddenNeuronCount;
    layer._neuronCount = neuronCount;

    this->_hiddenLayers.push_back(layer);
    this->_hiddenNeuronCount += neuronCount;
}

void NeuralNetwork::AllocateConnections() {
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
    size_t inputConnectionCount = 0;
    size_t outputConnectionCount = 0;
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

    outputConnectionCount += currentLayerOutputConnectionCount * this->_inputNeuronCount;

    size_t inputNeuronIndex = this->GetInputNeuronStartIndex();
    for (size_t i = 0; i < this->_inputNeuronCount; i++) {
        Neuron& neuron = this->_neurons[inputNeuronIndex + i];
        neuron._outputConnectionStartIndex = outputConnectionIndex;

        outputConnectionIndex += currentLayerOutputConnectionCount;
    }

    // The first bias neuron is the one for the input layer.
    Neuron& biasNeuron = this->_neurons[biasNeuronIndex++];
    biasNeuron._outputConnectionStartIndex = outputConnectionIndex;
    outputConnectionIndex = this->_hiddenLayers.front()._neuronCount;

    // Calculate the connections incoming to the output layer.
    if (this->_enableShortcutConnections) {
        // All input and hidden neurons are connected to each output neuron.
        currentLayerInputConnectionCount = this->_inputNeuronCount + this->_hiddenNeuronCount + 1;
    } else {
        // Output neurons are connected only to the last hidden layer.
        currentLayerInputConnectionCount = this->_hiddenLayers.back()._neuronCount + 1;
    }

    inputConnectionCount += currentLayerInputConnectionCount * this->_outputNeuronCount;

    size_t firstOutputNeuronIndex = this->GetOutputNeuronStartIndex();
    for (size_t i = 0; i < this->_outputNeuronCount; i++) {
        Neuron& neuron = this->_neurons[firstOutputNeuronIndex + i];
        neuron._inputConnectionStartIndex = inputConnectionIndex;

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
        outputConnectionIndex += biasOutputConnections;

        inputConnectionCount += currentLayer._neuronCount * currentLayerInputConnectionCount;
        outputConnectionCount += currentLayer._neuronCount * currentLayerOutputConnectionCount + biasOutputConnections;
    }

    this->_inputConnections.resize(inputConnectionCount);
    this->_outputConnections.resize(outputConnectionCount);
    this->_weights.resize(inputConnectionCount);
}

void NeuralNetwork::ConnectNeurons(size_t fromNeuronIndex, size_t toNeuronIndex) {
    Neuron& fromNeuron = this->_neurons[fromNeuronIndex];
    Neuron& toNeuron = this->_neurons[toNeuronIndex];

    size_t inputConnectionIndex = toNeuron._inputConnectionStartIndex + toNeuron._inputConnectionCount;
    size_t weightIndex = inputConnectionIndex;
    Connection& inputConnection = this->_inputConnections.at(inputConnectionIndex);
    inputConnection._neuron = fromNeuronIndex;
    inputConnection._weightIndex = weightIndex;
    toNeuron._inputConnectionCount++;

    size_t outputConnectionIndex = fromNeuron._outputConnectionStartIndex + fromNeuron._outputConnectionCount;
    Connection& outputConnection = this->_outputConnections.at(outputConnectionIndex);
    outputConnection._neuron = toNeuronIndex;
    outputConnection._weightIndex = weightIndex;
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
    this->AllocateConnections();
    this->ConnectFully();
}

void NeuralNetwork::Train(TrainingData* trainingData, size_t epochCount) {
    for (size_t i = 0; i < epochCount; i++) {
        this->_randomWrapper.ShuffleVector(trainingData);

        //this->ResetTrainingData(neuralNetwork);

        // Train the network using online weight updates - no batching
        for (size_t j = 0; j < trainingData->size(); j++)
        {
            // Run the network forward to get values in the output neurons.
            this->RunForward(&trainingData->at(j)._input);

            // Run the network backward to propagate the error values
            //neuralNetwork.RunBackward(*data.output.get(i));

            // Update weights online - no batching
            //this->UpdateWeights(neuralNetwork, data);
        }
    }
}

double NeuralNetwork::ExecuteActivationFunction(ActivationFunctionType activationFunctionType, double field) {
    return ActivationFunction::ExecuteSigmoidSymmetric(field);
}

void NeuralNetwork::ComputeNeuronValue(size_t neuronIndex) {
    Neuron& neuron = this->_neurons[neuronIndex];
    neuron._field = 0.0;

    // Sum incoming values.
    for (size_t i = 0; i < neuron._inputConnectionCount; i++) {
        size_t inputConnectionIndex = neuron._inputConnectionStartIndex + i;
        const Connection& connection = this->_inputConnections[inputConnectionIndex];
        const Neuron& fromNeuron = this->_neurons[connection._neuron];

        neuron._field += fromNeuron._value * this->_weights[connection._weightIndex];

        assert(inputConnectionIndex == connection._weightIndex);
    }

    neuron._value = ExecuteActivationFunction(neuron._activationFunctionType, neuron._field);
}

void NeuralNetwork::RunForward(std::vector<double>* input) {
    assert(input->size() == this->_inputNeuronCount);

    // Feed each input into the corresponding input neuron.
    size_t inputNeuronStartIndex = GetInputNeuronStartIndex();
    for (size_t i = 0; i < this->_inputNeuronCount; i++) {
        this->_neurons[inputNeuronStartIndex + i]._value = input->at(i);
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

void NeuralNetwork::InitializeWeightsRandom(double min, double max) {
    for (size_t i = 0; i < this->_weights.size(); i++) {
        this->_weights[i] = this->_randomWrapper.RandomFloat(min, max);
    }
}
