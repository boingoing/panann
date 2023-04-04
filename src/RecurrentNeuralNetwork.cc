//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#include <cassert>

#include "ActivationFunction.h"
#include "RecurrentNeuralNetwork.h"

namespace panann {

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
    layers_.emplace_back(cell_index, cell_count);

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

        cells_.emplace_back(cell_hidden_neuron_start_index,neurons_per_cell,cell_states_index,cell_memory_size);
    }
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

void RecurrentNeuralNetwork::FixNeuronConnectionIndices() {
    assert(GetCellLayerCount() > 0);
    assert(!IsTopologyConstructed());
    assert(AreNeuronsAllocated());
    assert(AreConnectionsAllocated());

    const auto& first_layer = layers_.front();
    const auto& last_layer = layers_.back();

    // Count of output connections from each of the input layer neurons.
    size_t input_layer_output_connection_count = 0;
    for (size_t i = 0; i < first_layer.cell_count; i++) {
        const size_t index = i + first_layer.cell_start_index;
        const auto& cell = GetCell(index);
        // All gates and layers except the output gate are connected to incoming connections to the cell.
        // That's the forget gate, input gate, candidate cell state layer, and hidden state layer.
        // Each gate contains |cell.cell_state_count| neurons so this cell contributes |4 * cell.cell_state_count| output connections for each input neuron.
        input_layer_output_connection_count += cell.cell_state_count * 4;
    }

    // Set the output connection start indices into the input layer neurons.
    for (size_t i = 0; i < GetInputNeuronCount(); i++) {
        auto& neuron = GetInputNeuron(i);
        neuron.output_connection_start_index = TakeOutputConnections(input_layer_output_connection_count);
    }

    // Count of input connections into each of the output layer neurons (+1 is for the bias connection).
    size_t output_layer_input_connection_count = 1;
    for (size_t i = 0; i < last_layer.cell_count; i++) {
        const size_t index = i + last_layer.cell_start_index;
        const auto& cell = GetCell(index);
        // Each cell in the last layer is connected to the output layer neurons.
        // The output layer in each cell is composed of |cell.cell_state_count| neurons.
        output_layer_input_connection_count += cell.cell_state_count;
    }

    // Set the input connection start indices into the output neurons.
    for (size_t i = 0; i < GetOutputNeuronCount(); i++) {
        auto& neuron = GetOutputNeuron(i);
        neuron.input_connection_start_index = TakeInputConnections(output_layer_input_connection_count);
    }

    // Set the input and output connection start indices into all the hidden neurons.
    for (size_t layer_index = 0; layer_index < GetHiddenLayerCount(); layer_index++) {
        // Each hidden neuron in the current layer is connected to the bias neuron in either the previous hidden layer or the input layer.
        size_t input_connection_count = 1;
        size_t output_connection_count = 0;

        if (enable_shortcut_connections_) {
            // All hidden layer neurons connect to the input layer when shortcuts are enabled.
            input_connection_count += GetInputNeuronCount();

            // Each neuron in this layer connects to the neurons in all previous hidden layers when shortcuts are enabled.
            for (size_t previous_layer_index = 0; previous_layer_index < layer_index; previous_layer_index++) {
                const auto& previous_layer = GetHiddenLayer(previous_layer_index);
                input_connection_count += previous_layer.neuron_count;
            }

            // All hidden layer neruons connect directly to the output layer when shortcuts are enabled.
            output_connection_count += GetOutputNeuronCount();

            // Each neuron in this layer connects to all neurons in subsequent hidden layers when shortcuts are enabled.
            for (size_t next_layer_index = layer_index + 1; next_layer_index < GetHiddenLayerCount(); next_layer_index++) {
                const auto& next_layer = GetHiddenLayer(next_layer_index);
                output_connection_count += next_layer.neuron_count;
            }
        } else {
            if (layer_index == 0) {
                // First hidden layer only connects to the input layer.
                input_connection_count += GetInputNeuronCount();
            } else {
                // Subsequent hidden layers connect directly the previous one.
                const auto& previous_layer = GetHiddenLayer(layer_index - 1);
                input_connection_count += previous_layer.neuron_count;
            }

            if (layer_index == GetHiddenLayerCount() - 1) {
                // Last hidden layer connects to the output layer.
                output_connection_count += GetOutputNeuronCount();
            } else {
                // This is not the last hidden layer, so there must be at least one more hidden layer after it.
                assert(layer_index + 1 < GetHiddenLayerCount());
                const auto& next_layer = GetHiddenLayer(layer_index + 1);

                // This hidden layer connects directly to the next one.
                output_connection_count += next_layer.neuron_count;
            }
        }

        // Save the start connection indices into the hidden neurons in the current layer.
        const auto& current_layer = GetHiddenLayer(layer_index);
        for (size_t i = 0; i < current_layer.neuron_count; i++) {
            auto& neuron = GetNeuron(current_layer.neuron_start_index + i);
            neuron.input_connection_start_index = TakeInputConnections(input_connection_count);
            neuron.output_connection_start_index = TakeOutputConnections(output_connection_count);
        }
    }

    // The first bias neuron connects to hidden neurons in the first layer.
    auto& first_bias_neuron = GetBiasNeuron(0);
    first_bias_neuron.output_connection_start_index = TakeOutputConnections(first_layer.neuron_count);

    // Set the output connection start indices into all but the last bias neuron - one per hidden layer.
    for (size_t layer_index = 0; layer_index < GetHiddenLayerCount() - 1; layer_index++) {
        // We're not walking to the last hidden layer so there always must be a next layer after current.
        assert(layer_index + 1 < GetHiddenLayerCount());
        const auto& next_layer = GetHiddenLayer(layer_index + 1);

        // The bias neuron for |layer_index| connects to each hidden neuron in the next hidden layer.
        // Bias neurons do not have incoming connections.
        auto& bias_neuron = GetBiasNeuron(1 + layer_index);
        bias_neuron.output_connection_start_index = TakeOutputConnections(next_layer.neuron_count);
    }

    // The last bias neuron connects to the output layer neurons.
    auto& last_bias_neuron = GetBiasNeuron(GetHiddenLayerCount());
    last_bias_neuron.output_connection_count = TakeOutputConnections(GetOutputNeuronCount());
}

void RecurrentNeuralNetwork::Allocate() {
    assert(!IsTopologyConstructed());

    cell_states_.resize(cell_states_count_);

    FeedForwardNeuralNetwork::Allocate();

    assert(GetHiddenLayerCount() > 0);

    // Allocate the neurons and memory cells

    // TODO(boingoing): Support memory cells with different memory sizes.
    const size_t neurons_per_gate = cell_memory_size_;

    // Forget gate, input gate, output gate, candidate cell state layer, hidden state layer
    const size_t neurons_per_cell = neurons_per_gate * 5;

    // hidden_neuron_count_ has the count of memory cells in the network.
    const size_t cell_count = GetHiddenNeuronCount();

    // Correct hidden_neuron_count_ to be an accurate count of hidden units in the network.
    // CONSIDER: Support ordinary hidden units as well as memory cells?
    hidden_neuron_count_ = neurons_per_cell * cell_count;

    // Input and output neurons + the neurons in each cell gate layer + bias one per cell and one for the output layer
    const size_t neuron_count =
        GetInputNeuronCount() +
        GetOutputNeuronCount() +
        GetHiddenNeuronCount() +
        1 + cell_count;

    this->neurons_.resize(neuron_count);
    this->cells_.resize(cell_count);
    this->cell_states_.resize(cell_count * this->cell_memory_size_);

    // Count all connections and save the starting connection index into the neurons / cells.

    size_t input_connection_index = 0;
    size_t output_connection_index = 0;

    // Per-neuron input connection count for the current layer.
    size_t current_layer_input_connection_count = 0;
    // Per-neuron output connection count for the current layer.
    size_t current_layer_output_connection_count = 0;

    // The input layer connects to all the neurons in each gate of the cells in the first hidden layer.
    const auto& first_layer = GetHiddenLayer(0);
    current_layer_output_connection_count = first_layer.neuron_count * neurons_per_gate * 4;

    // Count all connections outgoing from the input layer.
    for (size_t i = 0; i < GetInputNeuronCount(); i++) {
        Neuron& neuron = GetNeuron(GetInputNeuronStartIndex() + i);
        neuron.output_connection_start_index = output_connection_index;
        output_connection_index += current_layer_output_connection_count;
    }

    // Save the starting index for each cell's state memory.
    size_t cell_state_start_index = 0;
    // Count all the connections outgoing from the hidden layer bias neurons.
    size_t bias_neuron_index = GetBiasNeuronStartIndex();
    // Each cell has a bias neuron which is connected to all neurons in the cell.
    for (auto& cell : cells_) {
        cell.cell_state_start_index = cell_state_start_index;
        cell.cell_state_count = cell_memory_size_;

        auto& bias_neuron = GetNeuron(bias_neuron_index++);
        bias_neuron.output_connection_start_index = output_connection_index;
        bias_neuron.value = 1.0;
        output_connection_index += neurons_per_cell;
        cell_state_start_index += cell.cell_state_count;
    }

    // The output layer is also connected to a bias neuron.
    auto& bias_neuron_output = GetNeuron(bias_neuron_index++);
    bias_neuron_output.output_connection_start_index = output_connection_index;
    bias_neuron_output.value = 1.0;
    output_connection_index += GetOutputNeuronCount();

    // The output layer itself takes input from the output layer of each memory cell in the last hidden layer.
    const auto& last_layer = GetHiddenLayer(GetHiddenLayerCount()-1);
    current_layer_input_connection_count = last_layer.neuron_count * cell_memory_size_ + 1;
    for (size_t i = 0; i < GetOutputNeuronCount(); i++) {
        auto& neuron = GetNeuron(GetOutputNeuronStartIndex() + i);
        neuron.input_connection_start_index = input_connection_index;
        neuron.activation_function_type = GetOutputNeuronActivationFunctionType();
        input_connection_index += current_layer_input_connection_count;
    }

    // Now count the connections between all of the hidden layer cells.
    size_t hidden_neuron_index = GetHiddenNeuronStartIndex();
    size_t cell_index = 0;

    for (size_t layer_index = 0; layer_index < GetHiddenLayerCount(); layer_index++) {
        const auto& current_layer = GetHiddenLayer(layer_index);

        if (layer_index == 0) {
            // First layer of cells takes input from the input layer, previous hidden state, and a bias connection.
            current_layer_input_connection_count = GetInputNeuronCount() + cell_memory_size_ + 1;
        } else {
            const auto& previous_layer = GetHiddenLayer(layer_index - 1);
            // Remaining layers of cells take input from the output of previous layers, previous hidden state, and a bias connection.
            current_layer_input_connection_count = previous_layer.neuron_count * cell_memory_size_ + cell_memory_size_ + 1;
        }

        if (layer_index == GetHiddenLayerCount() - 1) {
            // Last layer of cells connect to the output layer.
            current_layer_output_connection_count = GetOutputNeuronCount();
        } else {
            assert(layer_index + 1 < GetHiddenLayerCount());

            const auto& next_layer = GetHiddenLayer(layer_index + 1);
            // Previous layers of cells connect to cell gates in subsequent layer.
            current_layer_output_connection_count = next_layer.neuron_count * cell_memory_size_ * 4;
        }

        // Cell output neurons also connect back to the gates of the cell.
        current_layer_output_connection_count += cell_memory_size_ * 4;

        // Count input connections and initialize all neurons in one gate of one cell.
        auto initialize_gate_neurons = [&](size_t neuron_start_index, size_t neurons_per_gate, ActivationFunctionType activation_function) {
            for (size_t k = 0; k < neurons_per_gate; k++) {
                auto& neuron = GetNeuron(neuron_start_index + k);
                neuron.input_connection_start_index = input_connection_index;
                // None of the neurons in the 4 gates of each cell have output connections.
                neuron.output_connection_start_index = 0;
                neuron.activation_function_type = activation_function;

                input_connection_index += current_layer_input_connection_count;
            }
        };

        // The hidden layer structure holds the count of cells in each layer, not the actual count of hidden units.
        for (size_t j = 0; j < current_layer.neuron_count; j++) {
            auto& cell = cells_[cell_index++];
            cell.neuron_start_index = hidden_neuron_index;
            cell.neuron_count = neurons_per_cell;

            // All the neurons in all 4 gates are connected to the same previous-layer output nodes.
            // Forget gate
            initialize_gate_neurons(hidden_neuron_index, neurons_per_gate, ActivationFunctionType::Sigmoid);
            hidden_neuron_index += neurons_per_gate;
            // Input gate
            initialize_gate_neurons(hidden_neuron_index, neurons_per_gate, ActivationFunctionType::Sigmoid);
            hidden_neuron_index += neurons_per_gate;
            // Candidate cell state
            initialize_gate_neurons(hidden_neuron_index, neurons_per_gate, ActivationFunctionType::SigmoidSymmetric);
            hidden_neuron_index += neurons_per_gate;
            // Output gate
            initialize_gate_neurons(hidden_neuron_index, neurons_per_gate, ActivationFunctionType::Sigmoid);
            hidden_neuron_index += neurons_per_gate;

            // The cell output units.
            for (size_t k = 0; k < neurons_per_gate; k++) {
                auto& neuron = GetNeuron(hidden_neuron_index++);
                // None of the neurons in the cell output layer have input connections.
                neuron.input_connection_start_index = 0;
                neuron.output_connection_start_index = output_connection_index;
                neuron.activation_function_type = ActivationFunctionType::SigmoidSymmetric;
                // The output units are recurrently connected to the gate neurons, we need to set a default value.
                neuron.value = 0.0;

                output_connection_index += current_layer_output_connection_count;
            }
        }
    }

    this->input_connections_.resize(input_connection_index);
    this->output_connections_.resize(output_connection_index);
    this->weights_.resize(input_connection_index);
}

void RecurrentNeuralNetwork::ConnectFully() {
    assert(!IsTopologyConstructed());
    assert(GetHiddenLayerCount() > 0);
    assert(!cells_.empty());

    size_t current_cell_index = 0;
    size_t bias_neuron_index = GetBiasNeuronStartIndex();
    const auto& first_layer = GetHiddenLayer(0);

    // Connect all cells in the first layer to the input neurons.
    for (size_t i = 0; i < first_layer.neuron_count; i++) {
        const auto& cell = cells_[current_cell_index++];

        // Connect input layer to gate neurons for this cell.
        ConnectLayers(GetInputNeuronStartIndex(),
            GetInputNeuronCount(),
            cell.neuron_start_index,
            cell.cell_state_count * 4);

        // Connect this cell output neurons to the gate neurons.
        ConnectLayers(cell.neuron_start_index + cell.cell_state_count * 4,
            cell.cell_state_count,
            cell.neuron_start_index,
            cell.cell_state_count * 4);

        // Connect this cell bias neuron to all neurons in the cell.
        ConnectBiasNeuron(bias_neuron_index++, cell.neuron_start_index, cell.neuron_count);
    }

    // Connect all cells in the subsequent layers to each other.
    for (size_t layer_index = 1; layer_index < GetHiddenLayerCount(); layer_index++) {
        const auto& previous_layer = GetHiddenLayer(layer_index - 1);
        const auto& current_layer = GetHiddenLayer(layer_index);

        for (size_t i = 0; i < current_layer.neuron_count; i++) {
            const auto& current_cell = cells_[current_layer.neuron_start_index + i];

            // Connect previous layer cells to the gate neurons for this cell.
            for (size_t j = 0; j < previous_layer.neuron_count; j++) {
                const auto& previous_cell = cells_[previous_layer.neuron_start_index + j];

                ConnectLayers(previous_cell.neuron_start_index + previous_cell.cell_state_count * 4,
                    previous_cell.cell_state_count,
                    current_cell.neuron_start_index,
                    current_cell.cell_state_count * 4);
            }

            // Connect this cell output neurons to the gate neurons.
            ConnectLayers(current_cell.neuron_start_index + current_cell.cell_state_count * 4,
                current_cell.cell_state_count,
                current_cell.neuron_start_index,
                current_cell.cell_state_count * 4);

            // Connect this cell bias neuron to all neurons in the cell.
            ConnectBiasNeuron(bias_neuron_index++, current_cell.neuron_start_index, current_cell.neuron_count);
        }
    }

    // The last layer or the only layer if there's only one hidden layer.
    const auto& last_layer = GetHiddenLayer(GetHiddenLayerCount() - 1);

    // Connect cells in the last hidden layer to the output layer.
    for (size_t i = 0; i < last_layer.neuron_count; i++) {
        const auto& cell = cells_[last_layer.neuron_start_index + i];

        ConnectLayers(cell.neuron_start_index + cell.cell_state_count * 4,
            cell.cell_state_count,
            GetOutputNeuronStartIndex(),
            GetOutputNeuronCount());
    }

    // Connect the output layer to its bias neuron.
    ConnectBiasNeuron(bias_neuron_index, GetOutputNeuronStartIndex(), GetOutputNeuronCount());
}

void RecurrentNeuralNetwork::UpdateCellState(const LongShortTermMemoryCell& cell) {
    const size_t forget_gate_neuron_start_index = cell.neuron_start_index;
    const size_t input_gate_neuron_start_index = forget_gate_neuron_start_index + cell.cell_state_count;
    const size_t candidate_cell_state_gate_neuron_start_index = input_gate_neuron_start_index + cell.cell_state_count;
    const size_t output_gate_neuron_start_index = candidate_cell_state_gate_neuron_start_index + cell.cell_state_count;
    const size_t cell_output_neuron_start = output_gate_neuron_start_index + cell.cell_state_count;

    ComputeNeuronValueRange(forget_gate_neuron_start_index, cell.cell_state_count);
    ComputeNeuronValueRange(input_gate_neuron_start_index, cell.cell_state_count);
    ComputeNeuronValueRange(candidate_cell_state_gate_neuron_start_index, cell.cell_state_count);
    ComputeNeuronValueRange(output_gate_neuron_start_index, cell.cell_state_count);

    for (size_t i = 0; i < cell.cell_state_count; i++) {
        const auto& forget_neuron = GetNeuron(forget_gate_neuron_start_index + i);
        const auto& input_neuron = GetNeuron(input_gate_neuron_start_index + i);
        const auto& candidate_state_neuron = GetNeuron(candidate_cell_state_gate_neuron_start_index + i);
        const auto& output_gate_neuron = GetNeuron(output_gate_neuron_start_index + i);
        auto& cell_output_neuron = GetNeuron(cell_output_neuron_start + i);
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
    for (auto& cell : cells_) {
        UpdateCellState(cell);
    }

    // Pull values into the output layer.
    const size_t output_neuron_start_index = GetOutputNeuronStartIndex();
    ComputeNeuronValueRange(output_neuron_start_index, GetOutputNeuronCount());
}

std::vector<double>& RecurrentNeuralNetwork::GetCellStates() {
    assert(IsTopologyConstructed());
    return cell_states_;
}

}  // namespace panann
