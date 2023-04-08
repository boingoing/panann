//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for
// full license information.
//-------------------------------------------------------------------------------------------------------

#include "MultiLayerNeuralTopology.h"

#include <algorithm>
#include <cassert>

#include "ActivationFunction.h"

namespace panann {

size_t MultiLayerNeuralTopology::GetHiddenLayerCount() const {
  return hidden_layers_.size();
}

MultiLayerNeuralTopology::Layer& MultiLayerNeuralTopology::GetHiddenLayer(
    size_t layer_index) {
  assert(layer_index < hidden_layers_.size());
  return hidden_layers_[layer_index];
}

const MultiLayerNeuralTopology::Layer& MultiLayerNeuralTopology::GetHiddenLayer(
    size_t layer_index) const {
  assert(layer_index < hidden_layers_.size());
  return hidden_layers_[layer_index];
}

void MultiLayerNeuralTopology::EnableShortcutConnections() {
  assert(!IsTopologyConstructed());
  enable_shortcut_connections_ = true;
}

void MultiLayerNeuralTopology::DisableShortcutConnections() {
  assert(!IsTopologyConstructed());
  enable_shortcut_connections_ = false;
}

void MultiLayerNeuralTopology::AddHiddenLayer(size_t neuron_count) {
  assert(!IsTopologyConstructed());

  // Update hidden neuron count to include the neurons newly added in this
  // layer. This returns the index where those neurons were added.
  const size_t starting_index = AddHiddenNeurons(neuron_count);

  // Add a new hidden layer beginning at the above hidden neuron index and
  // continuing for |neuron_count| neurons.
  hidden_layers_.emplace_back() = {starting_index, neuron_count};

  // Each hidden layer hooks-up to one bias neuron, keep track of the number of
  // bias neurons we need.
  AddBiasNeurons(1);
}

size_t MultiLayerNeuralTopology::GetInputConnectionCount() const {
  return input_connection_count_;
}

size_t MultiLayerNeuralTopology::GetOutputConnectionCount() const {
  return output_connection_count_;
}

size_t MultiLayerNeuralTopology::AddInputConnections(size_t count) {
  assert(!AreConnectionsAllocated());

  if (count == 0) {
    return 0;
  }

  const size_t index = input_connection_count_;
  input_connection_count_ += count;
  return index;
}

size_t MultiLayerNeuralTopology::AddOutputConnections(size_t count) {
  assert(!AreConnectionsAllocated());

  if (count == 0) {
    return 0;
  }

  const size_t index = output_connection_count_;
  output_connection_count_ += count;
  return index;
}

void MultiLayerNeuralTopology::FixNeuronConnectionIndices() {
  assert(GetHiddenLayerCount() > 0);
  assert(!IsTopologyConstructed());
  assert(AreNeuronsAllocated());
  assert(!AreConnectionsAllocated());

  const auto& first_layer = GetHiddenLayer(0);
  const auto& last_layer = GetHiddenLayer(GetHiddenLayerCount() - 1);

  // Count of output connections from each of the input neurons.
  const size_t input_layer_output_connection_count =
      enable_shortcut_connections_
          ? GetHiddenNeuronCount() + GetOutputNeuronCount()
          : first_layer.neuron_count;

  // Set the output connection start indices into the input neurons.
  for (size_t i = 0; i < GetInputNeuronCount(); i++) {
    auto& neuron = GetInputNeuron(i);
    // The input neuron must not have been assigned output connections already.
    assert(neuron.output_connection_start_index == 0);
    neuron.output_connection_start_index =
        AddOutputConnections(input_layer_output_connection_count);
  }

  // Count of input connections into each of the output neurons (+1 is for the
  // bias connection).
  const size_t output_layer_input_connection_count =
      1 + (enable_shortcut_connections_
               ? GetInputNeuronCount() + GetHiddenNeuronCount()
               : last_layer.neuron_count);

  // Set the input connection start indices into the output neurons.
  for (size_t i = 0; i < GetOutputNeuronCount(); i++) {
    auto& neuron = GetOutputNeuron(i);
    // The output neuron must not have been assigned input connections already.
    assert(neuron.input_connection_start_index == 0);
    neuron.input_connection_start_index =
        AddInputConnections(output_layer_input_connection_count);
  }

  // Set the input and output connection start indices into all the hidden
  // neurons.
  for (size_t layer_index = 0; layer_index < GetHiddenLayerCount();
       layer_index++) {
    // Each hidden neuron in the current layer is connected to a bias neuron.
    size_t input_connection_count = 1;
    size_t output_connection_count = 0;

    if (enable_shortcut_connections_) {
      // All hidden layer neurons connect to the input layer when shortcuts are
      // enabled.
      input_connection_count += GetInputNeuronCount();

      // Each neuron in this layer connects to the neurons in all previous
      // hidden layers when shortcuts are enabled.
      for (size_t previous_layer_index = 0; previous_layer_index < layer_index;
           previous_layer_index++) {
        const auto& previous_layer = GetHiddenLayer(previous_layer_index);
        input_connection_count += previous_layer.neuron_count;
      }

      // All hidden layer neruons connect directly to the output layer when
      // shortcuts are enabled.
      output_connection_count += GetOutputNeuronCount();

      // Each neuron in this layer connects to all neurons in subsequent hidden
      // layers when shortcuts are enabled.
      for (size_t next_layer_index = layer_index + 1;
           next_layer_index < GetHiddenLayerCount(); next_layer_index++) {
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
        // This is not the last hidden layer, so there must be at least one more
        // hidden layer after it.
        assert(layer_index + 1 < GetHiddenLayerCount());
        const auto& next_layer = GetHiddenLayer(layer_index + 1);

        // This hidden layer connects directly to the next one.
        output_connection_count += next_layer.neuron_count;
      }
    }

    // Save the start connection indices into the hidden neurons in the current
    // layer.
    const auto& current_layer = GetHiddenLayer(layer_index);
    for (size_t i = 0; i < current_layer.neuron_count; i++) {
      auto& neuron = GetNeuron(current_layer.neuron_start_index + i);
      // Verify the hidden neuron has not had connections assigned yet.
      assert(neuron.input_connection_start_index == 0);
      assert(neuron.output_connection_start_index == 0);
      neuron.input_connection_start_index =
          AddInputConnections(input_connection_count);
      neuron.output_connection_start_index =
          AddOutputConnections(output_connection_count);
    }
  }

  // Set the output connection start indices into the bias neurons for the
  // hidden layers.
  for (size_t layer_index = 0; layer_index < GetHiddenLayerCount();
       layer_index++) {
    // The bias neuron for |layer_index| connects to each hidden neuron in the
    // layer. Bias neurons do not have incoming connections.
    auto& bias_neuron = GetBiasNeuron(layer_index);
    assert(bias_neuron.output_connection_start_index == 0);

    const auto& layer = GetHiddenLayer(layer_index);
    bias_neuron.output_connection_start_index =
        AddOutputConnections(layer.neuron_count);
  }

  // The last bias neuron connects to the output layer neurons.
  auto& last_bias_neuron = GetBiasNeuron(GetHiddenLayerCount());
  assert(last_bias_neuron.output_connection_start_index == 0);
  last_bias_neuron.output_connection_start_index =
      AddOutputConnections(GetOutputNeuronCount());

  // Each input connection should have a matching output connection.
  assert(GetInputConnectionCount() == GetOutputConnectionCount());
}

void MultiLayerNeuralTopology::AllocateConnections() {
  assert(!AreConnectionsAllocated());

  input_connections_.resize(GetInputConnectionCount());
  output_connections_.resize(GetOutputConnectionCount());

  is_allocated_ = true;
}

bool MultiLayerNeuralTopology::AreConnectionsAllocated() const {
  return is_allocated_;
}

MultiLayerNeuralTopology::InputConnection&
MultiLayerNeuralTopology::GetInputConnection(size_t index) {
  assert(AreConnectionsAllocated());
  assert(index < input_connections_.size());
  return input_connections_[index];
}

MultiLayerNeuralTopology::OutputConnection&
MultiLayerNeuralTopology::GetOutputConnection(size_t index) {
  assert(AreConnectionsAllocated());
  assert(index < output_connections_.size());
  return output_connections_[index];
}

const MultiLayerNeuralTopology::InputConnection&
MultiLayerNeuralTopology::GetInputConnection(size_t index) const {
  assert(AreConnectionsAllocated());
  assert(index < input_connections_.size());
  return input_connections_[index];
}

const MultiLayerNeuralTopology::OutputConnection&
MultiLayerNeuralTopology::GetOutputConnection(size_t index) const {
  assert(AreConnectionsAllocated());
  assert(index < output_connections_.size());
  return output_connections_[index];
}

void MultiLayerNeuralTopology::ConnectNeurons(size_t from_neuron_index,
                                              size_t to_neuron_index) {
  auto& from_neuron = GetNeuron(from_neuron_index);
  auto& to_neuron = GetNeuron(to_neuron_index);

  // Each neuron has |input_connection_start_index| which is the index into the
  // input connections where the connections for this neuron are stored. All the
  // input connections for each neuron are stored sequentially starting at this
  // index. The neuron |to_neuron| has field |input_connection_count| which
  // contains the current count of input connections we've made to |to_neuron|.
  // |input_connection_index| is going to give us the next unassigned input
  // connection which was allocated for |to_neuron|.
  const size_t input_connection_index =
      to_neuron.input_connection_start_index + to_neuron.input_connection_count;
  auto& input_connection = GetInputConnection(input_connection_index);
  // The input connection we pulled from the end of the set for |to_neuron|
  // should be uninitialized.
  assert(input_connection.from_neuron_index == 0);
  assert(input_connection.to_neuron_index == 0);
  input_connection.from_neuron_index = from_neuron_index;
  input_connection.to_neuron_index = to_neuron_index;
  // We've assigned the above input connection, increment the counter stored in
  // the neuron so we can assign the next input connection next time we connect
  // to |to_neuron|.
  to_neuron.input_connection_count++;

  // Output connections work similarly to input connections. Each neuron has
  // |output_connection_start_index| which is the index into the output
  // connections array where the connections for this neuron are stored. All the
  // output connections for each neuron are stored sequentially starting at this
  // index. The neuron |from_neuron| has field |output_connection_count| which
  // contains the current count of output connections we've made from
  // |from_neuron|. |output_connection_index| is going to give us the next
  // unassigned output connection which was allocated for |from_neuron|.
  const size_t output_connection_index =
      from_neuron.output_connection_start_index +
      from_neuron.output_connection_count;
  auto& output_connection = GetOutputConnection(output_connection_index);
  // The output connection we pulled from the end of the set for |from_neuron|
  // should be uninitialized.
  assert(output_connection.input_connection_index == 0);
  output_connection.input_connection_index = input_connection_index;
  // We've assigned the above output connection, increment the counter stored in
  // the neuron so we can assign the next output connection next time we connect
  // from |from_neuron|.
  from_neuron.output_connection_count++;
}

void MultiLayerNeuralTopology::ConnectLayerToNeuron(size_t from_neuron_index,
                                                    size_t from_neuron_count,
                                                    size_t to_neuron_index) {
  for (size_t i = 0; i < from_neuron_count; i++) {
    ConnectNeurons(from_neuron_index + i, to_neuron_index);
  }
}

void MultiLayerNeuralTopology::ConnectLayers(size_t from_neuron_index,
                                             size_t from_neuron_count,
                                             size_t to_neuron_index,
                                             size_t to_neuron_count) {
  for (size_t i = 0; i < to_neuron_count; i++) {
    ConnectLayerToNeuron(from_neuron_index, from_neuron_count,
                         to_neuron_index + i);
  }
}

void MultiLayerNeuralTopology::ConnectBiasNeuron(size_t bias_neuron_index,
                                                 size_t to_neuron_index,
                                                 size_t to_neuron_count) {
  ConnectLayers(bias_neuron_index, 1, to_neuron_index, to_neuron_count);
}

void MultiLayerNeuralTopology::ConnectFully() {
  assert(!IsTopologyConstructed());
  assert(AreNeuronsAllocated());
  assert(AreConnectionsAllocated());
  assert(GetHiddenLayerCount() > 0);

  for (size_t layer_index = 0; layer_index < GetHiddenLayerCount();
       layer_index++) {
    const auto& current_layer = GetHiddenLayer(layer_index);

    if (enable_shortcut_connections_) {
      // Connect to input layer.
      ConnectLayers(GetInputNeuronStartIndex(), GetInputNeuronCount(),
                    current_layer.neuron_start_index,
                    current_layer.neuron_count);

      // Connect to all previous hidden layers.
      for (size_t prev_index = 0; prev_index < layer_index; prev_index++) {
        const auto& previous_layer = GetHiddenLayer(prev_index);
        ConnectLayers(
            previous_layer.neuron_start_index, previous_layer.neuron_count,
            current_layer.neuron_start_index, current_layer.neuron_count);
      }
    } else {
      if (layer_index == 0) {
        // Connect first hidden layer to input layer.
        ConnectLayers(GetInputNeuronStartIndex(), GetInputNeuronCount(),
                      current_layer.neuron_start_index,
                      current_layer.neuron_count);
      } else {
        // Connect to previous hidden layer.
        const auto& previous_layer = GetHiddenLayer(layer_index - 1);
        ConnectLayers(
            previous_layer.neuron_start_index, previous_layer.neuron_count,
            current_layer.neuron_start_index, current_layer.neuron_count);
      }
    }
  }

  // Connect output layer.
  if (enable_shortcut_connections_) {
    // Connect input layer to output layer.
    ConnectLayers(GetInputNeuronStartIndex(), GetInputNeuronCount(),
                  GetOutputNeuronStartIndex(), GetOutputNeuronCount());

    // Connect all hidden layers to output layer.
    for (size_t layer_index = 0; layer_index < GetHiddenLayerCount();
         layer_index++) {
      const auto& layer = GetHiddenLayer(layer_index);
      ConnectLayers(layer.neuron_start_index, layer.neuron_count,
                    GetOutputNeuronStartIndex(), GetOutputNeuronCount());
    }
  } else {
    const auto& last_layer = GetHiddenLayer(GetHiddenLayerCount() - 1);
    // Connect output layer to the last hidden layer.
    ConnectLayers(last_layer.neuron_start_index, last_layer.neuron_count,
                  GetOutputNeuronStartIndex(), GetOutputNeuronCount());
  }

  // Connect hidden layers to their bias neurons.
  for (size_t layer_index = 0; layer_index < GetHiddenLayerCount();
       layer_index++) {
    const auto& layer = GetHiddenLayer(layer_index);

    // Bias neurons do not have shortcut connections.
    // Just connect this layer to its bias neuron.
    ConnectBiasNeuron(GetBiasNeuronStartIndex() + layer_index,
                      layer.neuron_start_index, layer.neuron_count);
  }

  // Connect output layer to the last bias neuron.
  ConnectBiasNeuron(GetBiasNeuronStartIndex() + GetHiddenLayerCount(),
                    GetOutputNeuronStartIndex(), GetOutputNeuronCount());
}

void MultiLayerNeuralTopology::ConstructTopology() {
  assert(!IsTopologyConstructed());
  assert(!AreConnectionsAllocated());

  FixNeuronConnectionIndices();
  AllocateConnections();
  ConnectFully();

  is_constructed_ = true;
}

bool MultiLayerNeuralTopology::IsTopologyConstructed() const {
  return is_constructed_;
}

}  // namespace panann
