//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#include <cassert>

#include "RecurrentNeuralNetwork.h"
#include "ActivationFunction.h"

using namespace panann;

void RecurrentNeuralNetwork::SetCellMemorySize(size_t memorySize) {
    this->_cellMemorySize = memorySize;
}

size_t RecurrentNeuralNetwork::GetCellMemorySize() {
    return this->_cellMemorySize;
}

void RecurrentNeuralNetwork::Allocate() {
    assert(!this->_isConstructed);
    assert(!this->_hiddenLayers.empty());

    // Allocate the neurons and memory cells

    // TODO: Support memory cells with different memory sizes.
    size_t neuronsPerGate = this->_cellMemorySize;

    // Forget gate, input gate, output gate, candidate cell state layer, hidden state layer
    size_t neuronsPerCell = neuronsPerGate * 5;

    // _hiddenNeuronCount has the count of memory cells in the network.
    size_t cellCount = this->_hiddenNeuronCount;

    // Correct _hiddenNeuronCount to be an accurate count of hidden units in the network.
    // CONSIDER: Support ordinary hidden units as well as memory cells?
    this->_hiddenNeuronCount = neuronsPerCell * cellCount;

    // Input and output neurons + the neurons in each cell gate layer + bias one per cell and one for the output layer
    size_t neuronCount =
        this->_inputNeuronCount +
        this->_outputNeuronCount +
        this->_hiddenNeuronCount +
        1 + cellCount;

    this->_neurons.resize(neuronCount);
    this->_cells.resize(cellCount);

    // Count all connections and save the starting connection index into the neurons / cells.

    size_t inputConnectionIndex = 0;
    size_t outputConnectionIndex = 0;

    // Per-neuron input connection count for the current layer.
    size_t currentLayerInputConnectionCount = 0;
    // Per-neuron output connection count for the current layer.
    size_t currentLayerOutputConnectionCount = 0;

    // The input layer connects to all the neurons in each gate of the cells in the first hidden layer.
    currentLayerOutputConnectionCount = this->_hiddenLayers.front()._neuronCount * neuronsPerGate * 4;

    // Count all connections outgoing from the input layer.
    size_t inputNeuronIndex = this->GetInputNeuronStartIndex();
    for (size_t i = 0; i < this->_inputNeuronCount; i++) {
        Neuron& neuron = this->_neurons[inputNeuronIndex + i];
        neuron._outputConnectionStartIndex = outputConnectionIndex;
        outputConnectionIndex += currentLayerOutputConnectionCount;
    }

    // Count all the connections outgoing from the hidden layer bias neurons.
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

    // Now count the connections between all of the hidden layer cells.
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

            // Previous layers of cells connect to cell gates in subsequent layer.
            currentLayerOutputConnectionCount = this->_hiddenLayers[i + 1]._neuronCount * this->_cellMemorySize * 4;
        }

        // Cell output neurons also connect back to the gates of the cell.
        currentLayerOutputConnectionCount += this->_cellMemorySize * 4;

        // Count input connections and initialize all neurons in one gate of one cell.
        auto gateNeuronInitialize = [&](size_t neuronStartIndex, size_t neuronsPerGate, ActivationFunctionType activationFunction) {
            for (size_t k = 0; k < neuronsPerGate; k++) {
                Neuron& neuron = this->_neurons[neuronStartIndex + k];
                neuron._inputConnectionStartIndex = inputConnectionIndex;
                // None of the neurons in the 4 gates of each cell have output connections.
                neuron._outputConnectionStartIndex = 0;
                neuron._activationFunctionType = activationFunction;

                inputConnectionIndex += currentLayerInputConnectionCount;
            }
        };

        // The hidden layer structure holds the count of cells in each layer, not the actual count of hidden units.
        for (size_t j = 0; j < this->_hiddenLayers[i]._neuronCount; j++) {
            LongShortTermMemoryCell& cell = this->_cells[cellIndex++];
            cell._neuronStartIndex = hiddenNeuronIndex;
            cell._neuronCount = neuronsPerCell;

            // All the neurons in all 4 gates are connected to the same previous-layer output nodes.
            // Forget gate
            gateNeuronInitialize(hiddenNeuronIndex, neuronsPerGate, ActivationFunctionType::Sigmoid);
            hiddenNeuronIndex += neuronsPerGate;
            // Input gate
            gateNeuronInitialize(hiddenNeuronIndex, neuronsPerGate, ActivationFunctionType::Sigmoid);
            hiddenNeuronIndex += neuronsPerGate;
            // Candidate cell state
            gateNeuronInitialize(hiddenNeuronIndex, neuronsPerGate, ActivationFunctionType::SigmoidSymmetric);
            hiddenNeuronIndex += neuronsPerGate;
            // Output gate
            gateNeuronInitialize(hiddenNeuronIndex, neuronsPerGate, ActivationFunctionType::Sigmoid);
            hiddenNeuronIndex += neuronsPerGate;

            // The cell output units.
            for (size_t k = 0; k < neuronsPerGate; k++) {
                Neuron& neuron = this->_neurons[hiddenNeuronIndex++];
                // None of the neurons in the cell output layer have input connections.
                neuron._inputConnectionStartIndex = 0;
                neuron._outputConnectionStartIndex = outputConnectionIndex;
                neuron._activationFunctionType = ActivationFunctionType::SigmoidSymmetric;

                outputConnectionIndex += currentLayerOutputConnectionCount;
            }
        }
    }

    this->_inputConnections.resize(inputConnectionIndex);
    this->_outputConnections.resize(outputConnectionIndex);
}

void RecurrentNeuralNetwork::ConnectFully() {
    assert(!this->_isConstructed);
    assert(!this->_cells.empty());

    size_t currentCellIndex = 0;
    size_t inputNeuronStartIndex = this->GetInputNeuronStartIndex();
    size_t biasNeuronIndex = this->GetBiasNeuronStartIndex();

    // Connect all cells in the first layer to the input neurons.
    for (size_t i = 0; i < this->_hiddenLayers.front()._neuronCount; i++) {
        const LongShortTermMemoryCell& cell = this->_cells[currentCellIndex++];

        // Connect input layer to gate neurons for this cell.
        ConnectLayers(inputNeuronStartIndex,
            this->_inputNeuronCount,
            cell._neuronStartIndex,
            this->_cellMemorySize * 4); // TODO: Support per-cell memory size

        // Connect this cell output neurons to the gate neurons.
        ConnectLayers(cell._neuronStartIndex + this->_cellMemorySize * 4,
            this->_cellMemorySize,
            cell._neuronStartIndex,
            this->_cellMemorySize * 4);

        // Connect this cell bias neuron to all neurons in the cell.
        ConnectBiasNeuron(biasNeuronIndex++, cell._neuronStartIndex, cell._neuronCount);
    }

    // Connect all cells in the subsequent layers to each other.
    for (size_t layerIndex = 1; layerIndex < this->_hiddenLayers.size(); layerIndex++) {
        const Layer& previousLayer = this->_hiddenLayers[layerIndex - 1];
        const Layer& currentLayer = this->_hiddenLayers[layerIndex];

        for (size_t i = 0; i < currentLayer._neuronCount; i++) {
            const LongShortTermMemoryCell& currentCell = this->_cells[currentLayer._neuronStartIndex + i];

            // Connect previous layer cell to gate neurons for this cell.
            for (size_t j = 0; j < previousLayer._neuronCount; j++) {
                const LongShortTermMemoryCell& previousCell = this->_cells[previousLayer._neuronStartIndex + j];

                ConnectLayers(previousCell._neuronStartIndex + this->_cellMemorySize * 4,
                    this->_cellMemorySize,
                    currentCell._neuronStartIndex,
                    this->_cellMemorySize * 4); // TODO: Support per-cell memory size
            }

            // Connect this cell output neurons to the gate neurons.
            ConnectLayers(currentCell._neuronStartIndex + this->_cellMemorySize * 4,
                this->_cellMemorySize,
                currentCell._neuronStartIndex,
                this->_cellMemorySize * 4);

            // Connect this cell bias neuron to all neurons in the cell.
            ConnectBiasNeuron(biasNeuronIndex++, currentCell._neuronStartIndex, currentCell._neuronCount);
        }
    }

    size_t outputNeuronStartIndex = this->GetOutputNeuronStartIndex();
    const Layer& lastHiddenLayer = this->_hiddenLayers.back();

    // Connect last hidden layer to the output layer.
    for (size_t i = 0; i < lastHiddenLayer._neuronCount; i++) {
        const LongShortTermMemoryCell& cell = this->_cells[lastHiddenLayer._neuronStartIndex + i];

        ConnectLayers(cell._neuronStartIndex + this->_cellMemorySize * 4,
            this->_cellMemorySize,
            outputNeuronStartIndex,
            this->_outputNeuronCount);
    }

    // Connect the output layer to its bias neuron.
    ConnectBiasNeuron(biasNeuronIndex, outputNeuronStartIndex, this->_outputNeuronCount);
}

void RecurrentNeuralNetwork::Construct() {
    assert(!this->_isConstructed);
    assert(!this->_hiddenLayers.empty());

    this->Allocate();
    this->ConnectFully();
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
