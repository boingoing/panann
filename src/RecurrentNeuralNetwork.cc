//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#include <cassert>

#include "RecurrentNeuralNetwork.h"
#include "ActivationFunction.h"

using namespace panann;

void RecurrentNeuralNetwork::Construct() {
    assert(!this->_isConstructed);
    assert(!this->_hiddenLayers.empty());

    size_t neuronsPerGate = this->_cellMemorySize;

    // Forget gate, input gate, output gate, candidate cell state layer, hidden state layer
    size_t neuronsPerCell = neuronsPerGate * 5;

    // _hiddenNeuronCount has the count of memory cells in the network.
    size_t cellCount = this->_hiddenNeuronCount;

    // Correct _hiddenNeuronCount to be an accurate count of hidden units in the network.
    this->_hiddenNeuronCount = neuronsPerCell * cellCount;

    // Input and output neurons + the neurons in each cell gate layer + bias one per cell and one for the output layer
    size_t neuronCount =
        this->_inputNeuronCount +
        this->_outputNeuronCount +
        this->_hiddenNeuronCount +
        1 + cellCount;

    this->_neurons.resize(neuronCount);
    this->_cells.resize(cellCount);

    size_t inputConnectionIndex = 0;
    size_t outputConnectionIndex = 0;

    // Per-neuron input connection count for the current layer.
    size_t currentLayerInputConnectionCount = 0;
    // Per-neuron output connection count for the current layer.
    size_t currentLayerOutputConnectionCount = 0;

    // The input layer connects to all the neurons in each gate of the cells in the first hidden layer.
    currentLayerOutputConnectionCount = this->_hiddenLayers.front()._neuronCount * neuronsPerGate * 4;

    size_t inputNeuronIndex = this->GetInputNeuronStartIndex();
    for (size_t i = 0; i < this->_inputNeuronCount; i++) {
        Neuron& neuron = this->_neurons[inputNeuronIndex + i];
        neuron._outputConnectionStartIndex = outputConnectionIndex;
        outputConnectionIndex += currentLayerOutputConnectionCount;
    }

    size_t biasNeuronIndex = this->GetBiasNeuronStartIndex();
    // Each cell has a bias neuron which is connected to all neurons in the cell.
    for (size_t i = 0; i < cellCount; i++) {
        Neuron& biasNeuron = this->_neurons[biasNeuronIndex++];
        biasNeuron._outputConnectionStartIndex = outputConnectionIndex;
        biasNeuron._value = 1.0;
        outputConnectionIndex += neuronsPerCell;
    }

    // The output layer is also connected to a bias neuron.
    Neuron& biasNeuronOutput = this->_neurons[biasNeuronIndex++];
    biasNeuronOutput._outputConnectionStartIndex = outputConnectionIndex;
    biasNeuronOutput._value = 1.0;
    outputConnectionIndex += this->_outputNeuronCount;

    // The output layer itself takes input from the output layer of each memory cell in the last hidden layer.
    currentLayerInputConnectionCount = this->_hiddenLayers.back()._neuronCount * this->_cellMemorySize + 1;
    size_t outputNeuronIndex = this->GetOutputNeuronStartIndex();
    for (size_t i = 0; i < this->_outputNeuronCount; i++) {
        Neuron& neuron = this->_neurons[outputNeuronIndex + i];
        neuron._inputConnectionStartIndex = inputConnectionIndex;
        neuron._activationFunctionType = this->_outputNeuronActivationFunctionType;
        inputConnectionIndex += currentLayerInputConnectionCount;
    }

    size_t hiddenNeuronIndex = this->GetHiddenNeuronStartIndex();
    size_t cellIndex = 0;

    for (size_t i = 0; i < this->_hiddenLayers.size(); i++) {
        if (i == 0) {
            // First layer of cells takes input from the input layer, previous hidden state, and a bias connection.
            currentLayerInputConnectionCount = this->_inputNeuronCount + this->_cellMemorySize + 1;
        } else {
            // Remaining layers of cells take input from the output of previous layers, previous hidden state, and a bias connection.
            currentLayerInputConnectionCount = this->_hiddenLayers[i - 1]._neuronCount * this->_cellMemorySize + this->_cellMemorySize + 1;
        }

        if (i == this->_hiddenLayers.size() - 1) {
            // Last layer of cells connect to the output layer.
            currentLayerOutputConnectionCount = this->_outputNeuronCount;
        } else {
            assert(i + 1 < this->_hiddenLayers.size());

            // Previous layers of cells connect to subsequent hidden layers.
            currentLayerOutputConnectionCount = this->_hiddenLayers[i + 1]._neuronCount * this->_cellMemorySize * 4;
        }

        // The hidden layer structure holds the count of cells in each layer, not the actual count of hidden units.
        for (size_t j = 0; j < this->_hiddenLayers[i]._neuronCount; j++) {
            LongShortTermMemoryCell& cell = this->_cells[cellIndex++];
            cell._neuronStartIndex = hiddenNeuronIndex;
            cell._neuronCount = neuronsPerCell;

            // Forget, input, and output gates use sigmoid activation.
            // Candidate cell state is computed via tanh (sigmoid symmetric).
            ActivationFunctionType gateActivationFunctions[] = {
                ActivationFunctionType::Sigmoid,
                ActivationFunctionType::Sigmoid,
                ActivationFunctionType::SigmoidSymmetric,
                ActivationFunctionType::Sigmoid
            };

            // All the neurons in all 4 gates are connected to the same previous-layer output nodes.
            for (size_t k = 0; k < neuronsPerGate * 4; k++) {
                Neuron& neuron = this->_neurons[hiddenNeuronIndex + k];
                neuron._inputConnectionStartIndex = inputConnectionIndex;
                // None of the neurons in the 4 gates of each cell have output connections.
                neuron._outputConnectionStartIndex = 0;
                neuron._activationFunctionType = gateActivationFunctions[k % 4];

                inputConnectionIndex += currentLayerInputConnectionCount;
            }
            hiddenNeuronIndex += neuronsPerGate * 4;

            // The cell output units.
            for (size_t k = 0; k < neuronsPerGate; k++) {
                Neuron& neuron = this->_neurons[hiddenNeuronIndex + k];
                // None of the neurons in the cell output layer have input connections.
                neuron._inputConnectionStartIndex = 0;
                neuron._outputConnectionStartIndex = outputConnectionIndex;
                neuron._activationFunctionType = ActivationFunctionType::SigmoidSymmetric;

                outputConnectionIndex += currentLayerOutputConnectionCount;
            }
            hiddenNeuronIndex += neuronsPerGate;
        }
    }
}

void RecurrentNeuralNetwork::UpdateCellState(LongShortTermMemoryCell* cell) {
    size_t forgetWeightStart = 0;
    size_t forgetWeightCount = 0;
    size_t forgetNeuronStart = 0;
    size_t forgetNeuronCount = 0;
    size_t inputNeuronStart = 0;
    size_t inputNeuronCount = 0;

    for (size_t i = 0; i < inputNeuronCount; i++) {
        this->_neurons[forgetNeuronStart + i]._field =
            this->_neurons[inputNeuronStart + i]._value * this->_weights[forgetWeightStart + i];

        this->_neurons[forgetNeuronStart + i]._value = ActivationFunction::ExecuteSigmoid(this->_neurons[forgetNeuronStart + i]._field);
    }
}
