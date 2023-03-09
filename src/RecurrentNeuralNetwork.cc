//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#include <cassert>

#include "RecurrentNeuralNetwork.h"
#include "ActivationFunction.h"

using namespace panann;

RecurrentNeuralNetwork::RecurrentNeuralNetwork() :
    NeuralNetwork(),
    _cellMemorySize(200) {
}

void RecurrentNeuralNetwork::SetCellMemorySize(size_t memorySize) {
    this->_cellMemorySize = memorySize;
}

size_t RecurrentNeuralNetwork::GetCellMemorySize() {
    return this->_cellMemorySize;
}

void RecurrentNeuralNetwork::Allocate() {
    assert(!this->is_constructed_);
    assert(!this->hidden_layers_.empty());

    // Allocate the neurons and memory cells

    // TODO: Support memory cells with different memory sizes.
    size_t neuronsPerGate = this->_cellMemorySize;

    // Forget gate, input gate, output gate, candidate cell state layer, hidden state layer
    size_t neuronsPerCell = neuronsPerGate * 5;

    // hidden_neuron_count_ has the count of memory cells in the network.
    size_t cellCount = this->hidden_neuron_count_;

    // Correct hidden_neuron_count_ to be an accurate count of hidden units in the network.
    // CONSIDER: Support ordinary hidden units as well as memory cells?
    this->hidden_neuron_count_ = neuronsPerCell * cellCount;

    // Input and output neurons + the neurons in each cell gate layer + bias one per cell and one for the output layer
    size_t neuronCount =
        this->input_neuron_count_ +
        this->output_neuron_count_ +
        this->hidden_neuron_count_ +
        1 + cellCount;

    this->neurons_.resize(neuronCount);
    this->_cells.resize(cellCount);
    this->_cellStates.resize(cellCount * this->_cellMemorySize);

    // Count all connections and save the starting connection index into the neurons / cells.

    size_t inputConnectionIndex = 0;
    size_t outputConnectionIndex = 0;

    // Per-neuron input connection count for the current layer.
    size_t currentLayerInputConnectionCount = 0;
    // Per-neuron output connection count for the current layer.
    size_t currentLayerOutputConnectionCount = 0;

    // The input layer connects to all the neurons in each gate of the cells in the first hidden layer.
    currentLayerOutputConnectionCount = this->hidden_layers_.front().neuron_count * neuronsPerGate * 4;

    // Count all connections outgoing from the input layer.
    size_t inputNeuronIndex = this->GetInputNeuronStartIndex();
    for (size_t i = 0; i < this->input_neuron_count_; i++) {
        Neuron& neuron = this->neurons_[inputNeuronIndex + i];
        neuron.output_connection_start_index = outputConnectionIndex;
        outputConnectionIndex += currentLayerOutputConnectionCount;
    }

    // Save the starting index for each cell's state memory.
    size_t cellStateStartIndex = 0;
    // Count all the connections outgoing from the hidden layer bias neurons.
    size_t biasNeuronIndex = this->GetBiasNeuronStartIndex();
    // Each cell has a bias neuron which is connected to all neurons in the cell.
    for (size_t i = 0; i < cellCount; i++) {
        this->_cells[i]._cellStateStartIndex = cellStateStartIndex;

        Neuron& biasNeuron = this->neurons_[biasNeuronIndex++];
        biasNeuron.output_connection_start_index = outputConnectionIndex;
        biasNeuron.value = 1.0;
        outputConnectionIndex += neuronsPerCell;
        cellStateStartIndex += this->_cellMemorySize;
    }

    // The output layer is also connected to a bias neuron.
    Neuron& biasNeuronOutput = this->neurons_[biasNeuronIndex++];
    biasNeuronOutput.output_connection_start_index = outputConnectionIndex;
    biasNeuronOutput.value = 1.0;
    outputConnectionIndex += this->output_neuron_count_;

    // The output layer itself takes input from the output layer of each memory cell in the last hidden layer.
    currentLayerInputConnectionCount = this->hidden_layers_.back().neuron_count * this->_cellMemorySize + 1;
    size_t outputNeuronIndex = this->GetOutputNeuronStartIndex();
    for (size_t i = 0; i < this->output_neuron_count_; i++) {
        Neuron& neuron = this->neurons_[outputNeuronIndex + i];
        neuron.input_connection_start_index = inputConnectionIndex;
        neuron.activation_function_type = this->output_neuron_activation_function_type_;
        inputConnectionIndex += currentLayerInputConnectionCount;
    }

    // Now count the connections between all of the hidden layer cells.
    size_t hiddenNeuronIndex = this->GetHiddenNeuronStartIndex();
    size_t cellIndex = 0;

    for (size_t i = 0; i < this->hidden_layers_.size(); i++) {
        if (i == 0) {
            // First layer of cells takes input from the input layer, previous hidden state, and a bias connection.
            currentLayerInputConnectionCount = this->input_neuron_count_ + this->_cellMemorySize + 1;
        } else {
            // Remaining layers of cells take input from the output of previous layers, previous hidden state, and a bias connection.
            currentLayerInputConnectionCount = this->hidden_layers_[i - 1].neuron_count * this->_cellMemorySize + this->_cellMemorySize + 1;
        }

        if (i == this->hidden_layers_.size() - 1) {
            // Last layer of cells connect to the output layer.
            currentLayerOutputConnectionCount = this->output_neuron_count_;
        } else {
            assert(i + 1 < this->hidden_layers_.size());

            // Previous layers of cells connect to cell gates in subsequent layer.
            currentLayerOutputConnectionCount = this->hidden_layers_[i + 1].neuron_count * this->_cellMemorySize * 4;
        }

        // Cell output neurons also connect back to the gates of the cell.
        currentLayerOutputConnectionCount += this->_cellMemorySize * 4;

        // Count input connections and initialize all neurons in one gate of one cell.
        auto gateNeuronInitialize = [&](size_t neuronStartIndex, size_t neuronsPerGate, ActivationFunctionType activationFunction) {
            for (size_t k = 0; k < neuronsPerGate; k++) {
                Neuron& neuron = this->neurons_[neuronStartIndex + k];
                neuron.input_connection_start_index = inputConnectionIndex;
                // None of the neurons in the 4 gates of each cell have output connections.
                neuron.output_connection_start_index = 0;
                neuron.activation_function_type = activationFunction;

                inputConnectionIndex += currentLayerInputConnectionCount;
            }
        };

        // The hidden layer structure holds the count of cells in each layer, not the actual count of hidden units.
        for (size_t j = 0; j < this->hidden_layers_[i].neuron_count; j++) {
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
                Neuron& neuron = this->neurons_[hiddenNeuronIndex++];
                // None of the neurons in the cell output layer have input connections.
                neuron.input_connection_start_index = 0;
                neuron.output_connection_start_index = outputConnectionIndex;
                neuron.activation_function_type = ActivationFunctionType::SigmoidSymmetric;
                // The output units are recurrently connected to the gate neurons, we need to set a default value.
                neuron.value = 0.0;

                outputConnectionIndex += currentLayerOutputConnectionCount;
            }
        }
    }

    this->input_connections_.resize(inputConnectionIndex);
    this->output_connections_.resize(outputConnectionIndex);
    this->weights_.resize(inputConnectionIndex);
}

void RecurrentNeuralNetwork::ConnectFully() {
    assert(!this->is_constructed_);
    assert(!this->_cells.empty());

    size_t currentCellIndex = 0;
    size_t inputNeuronStartIndex = this->GetInputNeuronStartIndex();
    size_t biasNeuronIndex = this->GetBiasNeuronStartIndex();

    // Connect all cells in the first layer to the input neurons.
    for (size_t i = 0; i < this->hidden_layers_.front().neuron_count; i++) {
        const LongShortTermMemoryCell& cell = this->_cells[currentCellIndex++];

        // Connect input layer to gate neurons for this cell.
        ConnectLayers(inputNeuronStartIndex,
            this->input_neuron_count_,
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
    for (size_t layerIndex = 1; layerIndex < this->hidden_layers_.size(); layerIndex++) {
        const Layer& previousLayer = this->hidden_layers_[layerIndex - 1];
        const Layer& currentLayer = this->hidden_layers_[layerIndex];

        for (size_t i = 0; i < currentLayer.neuron_count; i++) {
            const LongShortTermMemoryCell& currentCell = this->_cells[currentLayer.neuron_start_index + i];

            // Connect previous layer cell to gate neurons for this cell.
            for (size_t j = 0; j < previousLayer.neuron_count; j++) {
                const LongShortTermMemoryCell& previousCell = this->_cells[previousLayer.neuron_start_index + j];

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
    const Layer& lastHiddenLayer = this->hidden_layers_.back();

    // Connect last hidden layer to the output layer.
    for (size_t i = 0; i < lastHiddenLayer.neuron_count; i++) {
        const LongShortTermMemoryCell& cell = this->_cells[lastHiddenLayer.neuron_start_index + i];

        ConnectLayers(cell._neuronStartIndex + this->_cellMemorySize * 4,
            this->_cellMemorySize,
            outputNeuronStartIndex,
            this->output_neuron_count_);
    }

    // Connect the output layer to its bias neuron.
    ConnectBiasNeuron(biasNeuronIndex, outputNeuronStartIndex, this->output_neuron_count_);
}

void RecurrentNeuralNetwork::Construct() {
    assert(!this->is_constructed_);
    assert(!this->hidden_layers_.empty());

    this->Allocate();
    this->ConnectFully();

    this->is_constructed_ = true;
}

void RecurrentNeuralNetwork::UpdateCellState(size_t cellIndex) {
    LongShortTermMemoryCell& cell = this->_cells[cellIndex];

    size_t cellStateStartIndex = cell._cellStateStartIndex;
    size_t cellNeuronStartIndex = cell._neuronStartIndex;
    size_t cellMemorySize = this->_cellMemorySize;
    size_t forgetGateNeuronStart = cellNeuronStartIndex;
    size_t inputGateNeuronStart = forgetGateNeuronStart + cellMemorySize;
    size_t candidateCellStateGateNeuronStart = inputGateNeuronStart + cellMemorySize;
    size_t outputGateNeuronStart = candidateCellStateGateNeuronStart + cellMemorySize;
    size_t cellOutputNeuronStart = outputGateNeuronStart + cellMemorySize;

    this->ComputeNeuronValueRange(forgetGateNeuronStart, cellMemorySize);
    this->ComputeNeuronValueRange(inputGateNeuronStart, cellMemorySize);
    this->ComputeNeuronValueRange(candidateCellStateGateNeuronStart, cellMemorySize);
    this->ComputeNeuronValueRange(outputGateNeuronStart, cellMemorySize);

    for (size_t i = 0; i < cellMemorySize; i++) {
        const Neuron& forgetNeuron = this->neurons_[forgetGateNeuronStart + i];
        const Neuron& inputNeuron = this->neurons_[inputGateNeuronStart + i];
        const Neuron& candidateStateNeuron = this->neurons_[candidateCellStateGateNeuronStart + i];
        const Neuron& outputGateNeuron = this->neurons_[outputGateNeuronStart + i];
        Neuron& cellOutputNeuron = this->neurons_[cellOutputNeuronStart + i];
        size_t cellStateIndex = cellStateStartIndex + i;

        // Modify cell state based on the gate and existing cell state values
        // Ct = ft * Ct-1 + it * C't

        this->_cellStates[cellStateIndex] = forgetNeuron.value * this->_cellStates[cellStateIndex] + inputNeuron.value * candidateStateNeuron.value;

        // Set the cell output values based on the output gate and cell state
        // ht = ot * tanh(Ct)
        cellOutputNeuron.value = outputGateNeuron.value * ActivationFunction::ExecuteSigmoidSymmetric(this->_cellStates[cellStateIndex]);
    }
}

void RecurrentNeuralNetwork::RunForward(const std::vector<double>* input) {
    assert(this->is_constructed_);
    assert(input->size() == this->input_neuron_count_);

    // Feed each input into the corresponding input neuron.
    size_t inputNeuronStartIndex = GetInputNeuronStartIndex();
    for (size_t i = 0; i < this->input_neuron_count_; i++) {
        this->neurons_[inputNeuronStartIndex + i].value = input->operator[](i);
    }

    for (size_t i = 0; i < this->_cells.size(); i++) {
        UpdateCellState(i);
    }

    // Pull values into the output layer.
    size_t outputNeuronStartIndex = GetOutputNeuronStartIndex();
    this->ComputeNeuronValueRange(outputNeuronStartIndex, this->output_neuron_count_);
}

std::vector<double>* RecurrentNeuralNetwork::GetCellStates() {
    assert(this->is_constructed_);

    return &this->_cellStates;
}
