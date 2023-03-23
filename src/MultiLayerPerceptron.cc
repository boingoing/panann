//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <cassert>

#include "ActivationFunction.h"
#include "MultiLayerPerceptron.h"

namespace panann {

size_t MultiLayerPerceptron::GetHiddenLayerCount() const {
    return hidden_layers_.size();
}

MultiLayerPerceptron::Layer& MultiLayerPerceptron::GetHiddenLayer(size_t layer_index) {
    assert(layer_index < hidden_layers_.size());
    return hidden_layers_[layer_index];
}

void MultiLayerPerceptron::EnableShortcutConnections() {
    assert(!is_constructed_);
    enable_shortcut_connections_ = true;
}

void MultiLayerPerceptron::DisableShortcutConnections() {
    assert(!is_constructed_);
    enable_shortcut_connections_ = false;
}

void MultiLayerPerceptron::SetHiddenNeuronActivationFunctionType(ActivationFunctionType type) {
    assert(!is_constructed_);
    hidden_neuron_activation_function_type_ = type;
}

ActivationFunctionType MultiLayerPerceptron::GetHiddenNeuronActivationFunctionType() const {
    return hidden_neuron_activation_function_type_;
}

void MultiLayerPerceptron::SetOutputNeuronActivationFunctionType(ActivationFunctionType type) {
    assert(!is_constructed_);
    output_neuron_activation_function_type_ = type;
}

ActivationFunctionType MultiLayerPerceptron::GetOutputNeuronActivationFunctionType() const {
    return output_neuron_activation_function_type_;
}

void MultiLayerPerceptron::AddHiddenLayer(size_t neuron_count) {
    assert(!IsConstructed());

    // Add a new hidden layer beginning at the current hidden neuron count and continuing for |neuron_count| neurons.
    size_t current_neuron_index = GetHiddenNeuronStartIndex() + GetHiddenNeuronCount();
    hidden_layers_.emplace_back() = {current_neuron_index, neuron_count};

    // Update hidden neuron count to include the neurons newly added in this layer.
    AddHiddenNeurons(neuron_count);

    // Each hidden layer hooks-up to one bias neuron, keep track of the number of bias neurons we need.
    AddBiasNeurons(1);
}

void MultiLayerPerceptron::Allocate() {
    assert(!IsConstructed());
    // Do not support networks with no hidden layers, no input neurons, or no output neurons.
    assert(GetHiddenLayerCount() > 0);
    assert(GetInputNeuronCount() > 0);
    assert(GetOutputNeuronCount() > 0);

    AllocateNeurons();

    size_t bias_neuron_index = GetBiasNeuronStartIndex();
    size_t input_connection_index = 0;
    size_t output_connection_index = 0;
    size_t current_layer_input_connection_count = 0;
    size_t current_layer_output_connection_count = 0;

    // Calculate the connections outgoing from the input layer.
    if (enable_shortcut_connections_) {
        // The input layer connects to all hidden layers and the output layer.
        current_layer_output_connection_count = GetHiddenNeuronCount() + GetOutputNeuronCount();
    } else {
        // The input layer connects only to the first hidden layer.
        current_layer_output_connection_count = hidden_layers_.front().neuron_count;
    }

    // Set the output connection indices into the input neurons.
    for (size_t i = 0; i < GetInputNeuronCount(); i++) {
        auto& neuron = GetInputNeuron(i);
        // |neuron| has |current_layer_output_connection_count| output connections beginning at |output_connection_index|.
        neuron.output_connection_start_index = output_connection_index;
        // The next free output connection begins where the set of output connections for the current neuron end.
        // Increment our index by that count.
        output_connection_index += current_layer_output_connection_count;
    }

    // The first bias neuron is the one for the input layer.
    // It has output connections to each hidden neuron.
    auto& bias_neuron_input = GetNeuron(bias_neuron_index++);
    bias_neuron_input.output_connection_start_index = output_connection_index;
    // TODO(boingoing): Should we set the bias neuron values someplace else?
    bias_neuron_input.value = 1.0;
    output_connection_index += hidden_layers_.front().neuron_count;

    // Calculate the connections incoming to the output layer.
    if (enable_shortcut_connections_) {
        // All input and hidden neurons are connected to each output neuron.
        current_layer_input_connection_count = GetInputNeuronCount() + GetHiddenNeuronCount() + 1;
    } else {
        // Output neurons are connected only to the last hidden layer.
        current_layer_input_connection_count = hidden_layers_.back().neuron_count + 1;
    }

    // Set the input connection indices into the output neurons.
    for (size_t i = 0; i < GetOutputNeuronCount(); i++) {
        auto& neuron = GetOutputNeuron(i);
        neuron.input_connection_start_index = input_connection_index;
        // TODO(boingoing): Should we set the activation function type elsewhere?
        neuron.activation_function_type = output_neuron_activation_function_type_;
        input_connection_index += current_layer_input_connection_count;
    }

    size_t neuron_index = GetHiddenNeuronStartIndex();

    // Calculate the connections to and from all hidden layers.
    for (size_t layer_index = 0; layer_index < hidden_layers_.size(); layer_index++) {
        current_layer_input_connection_count = 0;
        current_layer_output_connection_count = 0;

        if (enable_shortcut_connections_) {
            // All hidden layers connect to the input layer when shortcuts are enabled.
            current_layer_input_connection_count += GetInputNeuronCount() + 1;

            // Each neuron in this layer connects to the neurons in all previous hidden layers.
            for (size_t previous_layer_index = 0; previous_layer_index < layer_index; previous_layer_index++) {
                current_layer_input_connection_count += hidden_layers_[previous_layer_index].neuron_count + 1;
            }

            // All hidden layers connect directly to the output layer when shortcuts are enabled.
            current_layer_output_connection_count += GetOutputNeuronCount();

            // This layer connects to all neurons in subsequent hidden layers.
            for (size_t next_layer_index = layer_index + 1; next_layer_index < hidden_layers_.size(); next_layer_index++) {
                current_layer_output_connection_count += hidden_layers_[next_layer_index].neuron_count;
            }
        } else {
            if (layer_index == 0) {
                // First hidden layer connects to the input layer.
                current_layer_input_connection_count += GetInputNeuronCount() + 1;
            } else {
                // This hidden layer connects directly the previous one.
                current_layer_input_connection_count += hidden_layers_[layer_index - 1].neuron_count + 1;
            }

            if (layer_index == hidden_layers_.size() - 1) {
                // Last hidden layer connects to the output layer.
                current_layer_output_connection_count += GetOutputNeuronCount();
            } else {
                assert(layer_index + 1 < hidden_layers_.size());

                // This hidden layer connects directly to the next one.
                current_layer_output_connection_count += hidden_layers_[layer_index + 1].neuron_count;
            }
        }

        const auto& current_layer = hidden_layers_[layer_index];
        for (size_t i = 0; i < current_layer.neuron_count; i++) {
            auto& neuron = GetNeuron(neuron_index++);
            neuron.input_connection_start_index = input_connection_index;
            neuron.output_connection_start_index = output_connection_index;
            neuron.activation_function_type = hidden_neuron_activation_function_type_;

            input_connection_index += current_layer_input_connection_count;
            output_connection_index += current_layer_output_connection_count;
        }

        // Bias neurons cannot have shortcut connections.
        size_t bias_output_connection_count = 0;
        if (layer_index == hidden_layers_.size() - 1) {
            // Bias neuron in the last hidden layer connects to the output layer.
            bias_output_connection_count = GetOutputNeuronCount();
        } else {
            // Bias neuron in this hidden layer connects to the next hidden layer.
            bias_output_connection_count = hidden_layers_[layer_index + 1].neuron_count;
        }

        // Bias neurons do not have incoming connections.
        Neuron& bias_neuron = GetNeuron(bias_neuron_index++);
        bias_neuron.output_connection_start_index = output_connection_index;
        bias_neuron.value = 1.0;
        output_connection_index += bias_output_connection_count;
    }

    input_connections_.resize(input_connection_index);
    output_connections_.resize(output_connection_index);
    weights_.resize(input_connection_index);
}

void MultiLayerPerceptron::ConnectNeurons(size_t from_neuron_index, size_t to_neuron_index) {
    auto& from_neuron = GetNeuron(from_neuron_index);
    auto& to_neuron = GetNeuron(to_neuron_index);

    const size_t input_connection_index = to_neuron.input_connection_start_index + to_neuron.input_connection_count;
    auto& input_connection = input_connections_[input_connection_index];
    input_connection.from_neuron_index = from_neuron_index;
    input_connection.to_neuron_index = to_neuron_index;
    to_neuron.input_connection_count++;

    const size_t output_connection_index = from_neuron.output_connection_start_index + from_neuron.output_connection_count;
    auto& output_connection = output_connections_[output_connection_index];
    output_connection.input_connection_index = input_connection_index;
    from_neuron.output_connection_count++;
}

void MultiLayerPerceptron::ConnectLayerToNeuron(size_t from_neuron_index, size_t from_neuron_count, size_t to_neuron_index) {
    for (size_t i = 0; i < from_neuron_count; i++) {
        ConnectNeurons(from_neuron_index + i, to_neuron_index);
    }
}

void MultiLayerPerceptron::ConnectLayers(size_t from_neuron_index, size_t from_neuron_count, size_t to_neuron_index, size_t to_neuron_count) {
    for (size_t i = 0; i < to_neuron_count; i++) {
        ConnectLayerToNeuron(from_neuron_index, from_neuron_count, to_neuron_index + i);
    }
}

void MultiLayerPerceptron::ConnectBiasNeuron(size_t bias_neuron_index, size_t to_neuron_index, size_t to_neuron_count) {
    ConnectLayers(bias_neuron_index, 1, to_neuron_index, to_neuron_count);
}

void MultiLayerPerceptron::ConnectFully() {
    assert(!is_constructed_);
    assert(!hidden_layers_.empty());

    const size_t input_neuron_start_index = GetInputNeuronStartIndex();
    size_t bias_neuron_index = GetBiasNeuronStartIndex();

    for (auto iter = hidden_layers_.cbegin(); iter != hidden_layers_.cend(); iter++) {
        const auto& current_layer = *iter;

        if (enable_shortcut_connections_) {
            // Connect to input layer.
            ConnectLayers(input_neuron_start_index,
                          GetInputNeuronCount(),
                          current_layer.neuron_start_index,
                          current_layer.neuron_count);

            // Connect to all previous hidden layers.
            for (auto prev_iter = hidden_layers_.cbegin(); prev_iter != iter; prev_iter++) {
                const auto& previous_layer = *prev_iter;
                ConnectLayers(previous_layer.neuron_start_index,
                              previous_layer.neuron_count,
                              current_layer.neuron_start_index,
                              current_layer.neuron_count);
            }
        } else {
            if (iter == hidden_layers_.cbegin()) {
                // Connect first hidden layer to input layer.
                ConnectLayers(input_neuron_start_index,
                              GetInputNeuronCount(),
                              current_layer.neuron_start_index,
                              current_layer.neuron_count);
            } else {
                // Connect to previous hidden layer.
                const auto& previous_layer = *(iter-1);
                ConnectLayers(previous_layer.neuron_start_index,
                              previous_layer.neuron_count,
                              current_layer.neuron_start_index,
                              current_layer.neuron_count);
            }
        }

        // Bias neurons do not have shortcut connections.
        // Just connect this layer to the bias neuron in the layer before it.
        ConnectBiasNeuron(bias_neuron_index++, current_layer.neuron_start_index, current_layer.neuron_count);
    }

    const size_t output_neuron_start_index = GetOutputNeuronStartIndex();
    if (enable_shortcut_connections_) {
        // Connect input layer to output layer.
        ConnectLayers(input_neuron_start_index,
                      GetInputNeuronCount(),
                      output_neuron_start_index,
                      GetOutputNeuronCount());

        // Connect all hidden layers to output layer.
        for (const auto& layer : hidden_layers_) {
            ConnectLayers(layer.neuron_start_index,
                          layer.neuron_count,
                          output_neuron_start_index,
                          GetOutputNeuronCount());
        }
    } else {
        const Layer& previous_layer = hidden_layers_.back();
        // Connect output layer to the last hidden layer.
        ConnectLayers(previous_layer.neuron_start_index,
                      previous_layer.neuron_count,
                      output_neuron_start_index,
                      GetOutputNeuronCount());
    }

    // Connect output layer to the bias neuron in the last hidden layer.
    ConnectBiasNeuron(bias_neuron_index, output_neuron_start_index, GetOutputNeuronCount());
}

void MultiLayerPerceptron::Construct() {
    assert(!is_constructed_);

    Allocate();
    ConnectFully();

    is_constructed_ = true;
}

}  // namespace panann
