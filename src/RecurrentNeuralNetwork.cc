//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#include <cassert>
#include <cstdint>

#include "ActivationFunction.h"
#include "RecurrentNeuralNetwork.h"

namespace panann {

size_t RecurrentNeuralNetwork::LongShortTermMemoryCell::GetNeuronsPerGate() const {
    return cell_state_count;
}

size_t RecurrentNeuralNetwork::LongShortTermMemoryCell::GetForgetGateStartNeuronIndex() const {
    return neuron_start_index;
}

size_t RecurrentNeuralNetwork::LongShortTermMemoryCell::GetInputGateStartNeuronIndex() const {
    return neuron_start_index + cell_state_count;
}

size_t RecurrentNeuralNetwork::LongShortTermMemoryCell::GetOutputGateStartNeuronIndex() const {
    return neuron_start_index + cell_state_count * 2;

}

size_t RecurrentNeuralNetwork::LongShortTermMemoryCell::GetCandidateCellStateStartNeuronIndex() const {
    return neuron_start_index + cell_state_count * 3;

}

size_t RecurrentNeuralNetwork::LongShortTermMemoryCell::GetOutputUnitStartNeuronIndex() const {
    return neuron_start_index + cell_state_count * 4;
}

void RecurrentNeuralNetwork::SetCellMemorySize(size_t memory_size) {
    cell_memory_size_ = memory_size;
}

size_t RecurrentNeuralNetwork::GetCellMemorySize() const {
    return cell_memory_size_;
}

size_t RecurrentNeuralNetwork::AddCellMemoryStates(size_t count) {
    assert(!IsTopologyConstructed());

    const size_t index = cell_states_count_;
    cell_states_count_ += count;
    return index;
}

void RecurrentNeuralNetwork::AddHiddenLayer(size_t cell_count, const std::vector<size_t>& cell_memory_sizes /* = {} */) {
    const size_t cell_index = cells_.size();
    layers_.emplace_back() = {cell_index, cell_count};

    // Each LSTM cell may have a different number of memory states so we can't just track how many cells there are in the network - we need to remember how many memory states each of those cells is holding.
    for (size_t i = 0; i < cell_count; i++) {
        // If |cell_memory_sizes| doesn't contain an entry for |i| or that entry is zero, use the default value provided by GetCellMemorySize().
        const size_t cell_memory_size = cell_memory_sizes.size() <= i || cell_memory_sizes[i] == 0 ? GetCellMemorySize() : cell_memory_sizes[i];
        // Just another name for the above. There are |cell_memory_size| neurons in each gate.
        const size_t neurons_per_gate = cell_memory_size;
        // Total count of neurons making up the cell: forget gate, input gate, output gate, candidate cell state layer, hidden state layer.
        const size_t neurons_per_cell = neurons_per_gate * 5;
        // Add hidden neurons needed for the structure of the cell. This returns the index at which those new neurons begin.
        const size_t cell_hidden_neuron_start_index = AddHiddenNeurons(neurons_per_cell);
        // Add cell memory states needed for the cell. This also returns the index at which those new memory states begin.
        const size_t cell_states_index = AddCellMemoryStates(cell_memory_size);
        // Each cell contributes one bias neuron to the network.
        AddBiasNeurons(1);

        cells_.emplace_back() = {cell_hidden_neuron_start_index,neurons_per_cell,cell_states_index,cell_memory_size};
    }
}

size_t RecurrentNeuralNetwork::GetCellCount() const {
    return cells_.size();
}

RecurrentNeuralNetwork::LongShortTermMemoryCell& RecurrentNeuralNetwork::GetCell(size_t index) {
    assert(index < cells_.size());
    return cells_[index];
}

size_t RecurrentNeuralNetwork::GetCellLayerCount() const {
    return layers_.size();
}

RecurrentNeuralNetwork::CellLayer& RecurrentNeuralNetwork::GetCellLayer(size_t index) {
    assert(index < layers_.size());
    return layers_[index];
}

void RecurrentNeuralNetwork::InitializeCellNeuronsOneGate(size_t neuron_start_index, size_t neurons_per_gate, ActivationFunctionType activation_function, size_t input_connection_count, size_t output_connection_count) {
    for (size_t i = 0; i < neurons_per_gate; i++) {
        auto& neuron = GetNeuron(neuron_start_index + i);
        // The neuron must not have been already assigned.
        assert(neuron.input_connection_start_index == 0);
        assert(neuron.output_connection_start_index == 0);
        neuron.input_connection_start_index = TakeInputConnections(input_connection_count);
        neuron.output_connection_start_index = TakeOutputConnections(output_connection_count);
        neuron.activation_function_type = activation_function;
        // TODO(boingoing): Is this still necessary?
        // The output units are recurrently connected to the gate neurons, we need to set a default value.
        neuron.value = 0.0;
    }
}

void RecurrentNeuralNetwork::InitializeCellNeurons(const LongShortTermMemoryCell& cell, size_t input_connection_count, size_t output_connection_count) {
    // All the neurons in all 4 gates are connected to the same previous-layer output nodes - which should be |input_connection_count| input connections.
    // None of the neurons in the 4 gates of each cell have output connections.
    // Forget gate
    InitializeCellNeuronsOneGate(cell.GetForgetGateStartNeuronIndex(), cell.GetNeuronsPerGate(), ActivationFunctionType::Sigmoid, input_connection_count, 0);
    // Input gate
    InitializeCellNeuronsOneGate(cell.GetInputGateStartNeuronIndex(), cell.GetNeuronsPerGate(), ActivationFunctionType::Sigmoid, input_connection_count, 0);
    // Candidate cell state
    InitializeCellNeuronsOneGate(cell.GetCandidateCellStateStartNeuronIndex(), cell.GetNeuronsPerGate(), ActivationFunctionType::SigmoidSymmetric, input_connection_count, 0);
    // Output gate
    InitializeCellNeuronsOneGate(cell.GetOutputGateStartNeuronIndex(), cell.GetNeuronsPerGate(), ActivationFunctionType::Sigmoid, input_connection_count, 0);

    // The cell output units are connected to the gates of cells in the next layer and recurrently connected to the gate neurons - which should be |output_connection_count| output connections.
    // None of the output unit neurons have input connections.
    InitializeCellNeuronsOneGate(cell.GetOutputUnitStartNeuronIndex(), cell.GetNeuronsPerGate(), ActivationFunctionType::SigmoidSymmetric, 0, output_connection_count);
}

void RecurrentNeuralNetwork::FixNeuronConnectionIndices() {
    assert(GetCellLayerCount() > 0);
    assert(!IsTopologyConstructed());
    assert(AreNeuronsAllocated());
    assert(AreConnectionsAllocated());

    const auto& first_layer = GetCellLayer(0);
    const auto& last_layer = GetCellLayer(GetCellLayerCount() - 1);

    // The input layer connects to all the neurons in each gate of the cells in the first hidden layer.
    size_t input_layer_output_connection_count = 0;
    for (size_t i = 0; i < first_layer.cell_count; i++) {
        const size_t index = i + first_layer.cell_start_index;
        const auto& cell = GetCell(index);
        // All gates and layers except the output gate are connected to incoming connections to the cell.
        // That's the forget gate, input gate, candidate cell state layer, and hidden state layer.
        // Each gate contains |cell.cell_state_count| neurons so this cell contributes |4 * cell.cell_state_count| output connections for each input neuron.
        input_layer_output_connection_count += cell.GetNeuronsPerGate() * 4;
    }

    // Set the output connection start indices into the input layer neurons.
    for (size_t i = 0; i < GetInputNeuronCount(); i++) {
        auto& neuron = GetInputNeuron(i);
        neuron.output_connection_start_index = TakeOutputConnections(input_layer_output_connection_count);
    }

    // The output layer takes input from the output layer of each memory cell in the last hidden layer (+1 for the bias connection).
    size_t output_layer_input_connection_count = 1;
    for (size_t i = 0; i < last_layer.cell_count; i++) {
        const size_t index = i + last_layer.cell_start_index;
        const auto& cell = GetCell(index);
        // Each cell in the last layer is connected to the output layer neurons.
        // The output layer in each cell is composed of |cell.cell_state_count| neurons.
        output_layer_input_connection_count += cell.GetNeuronsPerGate();
    }

    // Set the input connection start indices into the output neurons.
    for (size_t i = 0; i < GetOutputNeuronCount(); i++) {
        auto& neuron = GetOutputNeuron(i);
        neuron.input_connection_start_index = TakeInputConnections(output_layer_input_connection_count);
    }

    // Set the input and output connection start indices into all of the cell neurons.
    for (size_t layer_index = 0; layer_index < GetCellLayerCount(); layer_index++) {
        const auto& current_layer = GetCellLayer(layer_index);
        // Each gate neuron in the current cell layer is connected to a bias neuron.
        size_t input_connection_count = 1;
        size_t output_connection_count = 0;

        if (layer_index == 0) {
            // First layer of cells takes input from the input layer.
            input_connection_count += GetInputNeuronCount();
        } else {
            // Remaining layers of cells take input from the output of the previous layer.
            const auto& previous_layer = GetCellLayer(layer_index - 1);
            // Each cell may have a different memory size so we need to walk through the cells in the previous layer and count how many connections each will contribute.
            for (size_t i = 0; i < previous_layer.cell_count; i++) {
                const size_t cell_index = previous_layer.cell_start_index + i;
                const auto& cell = GetCell(cell_index);
                input_connection_count += cell.GetNeuronsPerGate();
            }
        }

        if (layer_index == GetCellLayerCount() - 1) {
            // Last layer of cells are connected to the output layer.
            output_connection_count = GetOutputNeuronCount();
        } else {
            // Current layer cells connect to all cell gates in the next layer.
            const auto& next_layer = GetCellLayer(layer_index + 1);
            // Each cell may have a different memory size so we need to walk through the cells in the next layer and count how many connections each will contribute.
            for (size_t i = 0; i < next_layer.cell_count; i++) {
                const size_t cell_index = next_layer.cell_start_index + i;
                const auto& cell = GetCell(cell_index);
                output_connection_count += cell.GetNeuronsPerGate() * 4;
            }
        }

        // Initialize all the neurons making up each cell in the current layer.
        for (size_t i = 0; i < current_layer.cell_count; i++) {
            const size_t cell_index = current_layer.cell_start_index + i;
            const auto& cell = GetCell(cell_index);
            // Each gate of the cell is also recurrently connected to the previous hidden state of the cell.
            const size_t cell_input_connection_count = input_connection_count + cell.GetNeuronsPerGate();
            // Cell output units also connect back to the gates of the cell.
            const size_t cell_output_connection_count = output_connection_count + cell.GetNeuronsPerGate() * 4;
            InitializeCellNeurons(cell, cell_input_connection_count, cell_output_connection_count);
        }
    }

    // Each cell has a bias neuron which is connected to all neurons in the cell.
    for (size_t i = 0; i < GetCellCount(); i++) {
        auto& neuron = GetBiasNeuron(i);
        const auto& cell = GetCell(i);
        neuron.output_connection_start_index = TakeOutputConnections(cell.neuron_count);
    }

    // The last bias neuron connects to the output layer neurons.
    auto& last_bias_neuron = GetBiasNeuron(GetCellCount());
    last_bias_neuron.output_connection_start_index = TakeOutputConnections(GetOutputNeuronCount());
}

void RecurrentNeuralNetwork::AllocateCellStates() {
    assert(!AreCellStatesAllocated());
    cell_states_.resize(cell_states_count_);
    is_allocated_ = true;
}

bool RecurrentNeuralNetwork::AreCellStatesAllocated() const {
    return is_allocated_;
}

void RecurrentNeuralNetwork::ConnectFully() {
    assert(!IsTopologyConstructed());
    assert(AreNeuronsAllocated());
    assert(GetCellLayerCount() > 0);

    const auto& first_layer = GetCellLayer(0);

    // Connect all cells in the first layer to the input neurons.
    for (size_t i = 0; i < first_layer.cell_count; i++) {
        const size_t cell_index = first_layer.cell_start_index + i;
        const auto& cell = GetCell(cell_index);

        // Connect input layer to gate neurons for this cell.
        ConnectLayers(GetInputNeuronStartIndex(),
            GetInputNeuronCount(),
            cell.neuron_start_index,
            cell.GetNeuronsPerGate() * 4);

        // Recurrently connect this cell output units to the gate neurons.
        ConnectLayers(cell.GetOutputUnitStartNeuronIndex(),
            cell.GetNeuronsPerGate(),
            cell.neuron_start_index,
            cell.GetNeuronsPerGate() * 4);

        // Connect this cell bias neuron to all neurons in the cell.
        ConnectBiasNeuron(cell_index, cell.neuron_start_index, cell.neuron_count);
    }

    // Connect all cells in the subsequent layers to each other.
    for (size_t layer_index = 1; layer_index < GetCellLayerCount(); layer_index++) {
        const auto& previous_layer = GetCellLayer(layer_index - 1);
        const auto& current_layer = GetCellLayer(layer_index);

        for (size_t i = 0; i < current_layer.cell_count; i++) {
            const size_t cell_index = current_layer.cell_start_index + i;
            const auto& current_cell = GetCell(cell_index);

            // Connect previous layer cells to the gate neurons for this cell.
            for (size_t j = 0; j < previous_layer.cell_count; j++) {
                const auto& previous_cell = GetCell(previous_layer.cell_start_index + j);

                ConnectLayers(previous_cell.GetOutputUnitStartNeuronIndex(),
                    previous_cell.GetNeuronsPerGate(),
                    current_cell.neuron_start_index,
                    current_cell.GetNeuronsPerGate() * 4);
            }

            // Recurrently connect this cell output units to the gate neurons.
            ConnectLayers(current_cell.GetOutputUnitStartNeuronIndex(),
                current_cell.GetNeuronsPerGate(),
                current_cell.neuron_start_index,
                current_cell.GetNeuronsPerGate() * 4);

            // Connect this cell bias neuron to all neurons in the cell.
            ConnectBiasNeuron(cell_index, current_cell.neuron_start_index, current_cell.neuron_count);
        }
    }

    // The last layer or the only layer if there's only one layer of cells.
    const auto& last_layer = GetCellLayer(GetCellLayerCount() - 1);

    // Connect cells in the last hidden layer to the output layer.
    for (size_t i = 0; i < last_layer.cell_count; i++) {
        const size_t cell_index = last_layer.cell_start_index + i;
        const auto& cell = GetCell(cell_index);

        ConnectLayers(cell.GetOutputUnitStartNeuronIndex(),
            cell.GetNeuronsPerGate(),
            GetOutputNeuronStartIndex(),
            GetOutputNeuronCount());
    }

    // Connect the output layer to the last bias neuron.
    ConnectBiasNeuron(GetCellLayerCount(), GetOutputNeuronStartIndex(), GetOutputNeuronCount());
}

void RecurrentNeuralNetwork::UpdateCellState(const LongShortTermMemoryCell& cell) {
    ComputeNeuronValueRange(cell.GetForgetGateStartNeuronIndex(), cell.GetNeuronsPerGate());
    ComputeNeuronValueRange(cell.GetInputGateStartNeuronIndex(), cell.GetNeuronsPerGate());
    ComputeNeuronValueRange(cell.GetCandidateCellStateStartNeuronIndex(), cell.GetNeuronsPerGate());
    ComputeNeuronValueRange(cell.GetOutputGateStartNeuronIndex(), cell.GetNeuronsPerGate());

    for (size_t i = 0; i < cell.GetNeuronsPerGate(); i++) {
        const auto& forget_neuron = GetNeuron(cell.GetForgetGateStartNeuronIndex() + i);
        const auto& input_neuron = GetNeuron(cell.GetInputGateStartNeuronIndex() + i);
        const auto& candidate_state_neuron = GetNeuron(cell.GetCandidateCellStateStartNeuronIndex() + i);
        const auto& output_gate_neuron = GetNeuron(cell.GetOutputGateStartNeuronIndex() + i);
        auto& cell_output_neuron = GetNeuron(cell.GetOutputUnitStartNeuronIndex() + i);
        const size_t cell_state_index = cell.cell_state_start_index + i;

        // Modify cell state based on the gate and existing cell state values
        // Ct = ft * Ct-1 + it * C't
        cell_states_[cell_state_index] = forget_neuron.value * cell_states_[cell_state_index] + input_neuron.value * candidate_state_neuron.value;

        // Set the cell output values based on the output gate and cell state
        // ht = ot * tanh(Ct)
        cell_output_neuron.value = output_gate_neuron.value * ActivationFunction::ExecuteSigmoidSymmetric(cell_states_[cell_state_index]);
    }
}

void RecurrentNeuralNetwork::RunForward(const std::vector<double>& input) {
    assert(IsTopologyConstructed());
    assert(input.size() == GetInputNeuronCount());

    // Feed each input into the corresponding input neuron.
    for (size_t i = 0; i < GetInputNeuronCount(); i++) {
        auto& neuron = GetInputNeuron(i);
        neuron.value = input[i];
    }

    // Update cell states.
    for (const auto& cell : cells_) {
        UpdateCellState(cell);
    }

    // Pull values into the output layer.
    const size_t output_neuron_start_index = GetOutputNeuronStartIndex();
    ComputeNeuronValueRange(output_neuron_start_index, GetOutputNeuronCount());
}

std::vector<double>& RecurrentNeuralNetwork::GetCellStates() {
    assert(AreCellStatesAllocated());
    return cell_states_;
}

}  // namespace panann
