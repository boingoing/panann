//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#ifndef MULTILAYERTOPOLOGY_H__
#define MULTILAYERTOPOLOGY_H__

#include <vector>

#include "NeuronContainer.h"

namespace panann {

/**
 * Supports building multi-layer topologies made up of neurons.<br/>
 * This class is intended only to facilitate building the structure of networks built from neurons and organized into layers.<br/>
 * To that end, it knows how to allocate and assign connections between all the neurons in each layer.<br/>
 * 
 * Primarily useful to group hidden neurons into layers and track input and output connections to and from each neuron in the topology.<br/>
 */
class MultiLayerNeuralTopology : public NeuronContainer {
protected:
    struct Layer {
        /**
         * Index at which the neurons comprising this layer begin.
        */
        size_t neuron_start_index;

        /**
         * Count of hidden neurons in this layer.
        */
        size_t neuron_count;
    };

    struct InputConnection {
        size_t from_neuron_index;
        size_t to_neuron_index;
    };

    struct OutputConnection {
        size_t input_connection_index;
    };

public:
    MultiLayerNeuralTopology() = default;
    MultiLayerNeuralTopology(const MultiLayerNeuralTopology&) = delete;
    MultiLayerNeuralTopology& operator=(const MultiLayerNeuralTopology&) = delete;
    ~MultiLayerNeuralTopology() override = default;

    /**
     * Append a hidden layer to the end of the list of existing hidden layers.<br/>
     * Hidden layers are located after the input layer and before the output layer.<br/>
     * Once added, hidden layers may not be removed.<br/>
     * Hidden layers may not be added after the network topology has been constructed.
     */
    void AddHiddenLayer(size_t neuron_count);

    /**
     * Shortcut connections are feed-forward connections between two non-adjacent layers.<br/>
     * Note: Changing this setting after the network topology has been constructed will have no impact on the network topology.<br/>
     * Default: disabled
     */
    void EnableShortcutConnections();
    void DisableShortcutConnections();

    /**
     * Build the network topology.<br/>
     * After construction, the number of input and output neurons, number of hidden layers, use of shortcut connections, and some other settings may not be modified.
     */
    void ConstructTopology();

    /**
     * Returns true if the network topology has been constructed and false otherwise.<br/>
     * Note: Once constructed, the network topology is fixed and cannot be changed.
     */
    bool IsTopologyConstructed() const;

protected:
    /**
     * Get the count of hidden layers in the topology.
     */
    size_t GetHiddenLayerCount() const;

    /**
     * Get a writable view of the hidden layer at |layer_index|.
     */
    Layer& GetHiddenLayer(size_t layer_index);

    /**
     * Get a read-only view of the hidden layer at |layer_index|.
     */
    const Layer& GetHiddenLayer(size_t layer_index) const;

    /**
     * Get the count of all input connections in the network topology.
     */
    size_t GetInputConnectionCount() const;

    /**
     * Get the count of all output connections in the network topology.
     */
    size_t GetOutputConnectionCount() const;

    /**
     * Get a writable view of a single input connection at |index| in the |input_connections_| vector.
     */
    InputConnection& GetInputConnection(size_t index);

    /**
     * Get a writable view of a single output connection at |index| in the |output_connections_| vector.
     */
    OutputConnection& GetOutputConnection(size_t index);

    /**
     * Assign the next |count| input connections and return the index at which these connections begin.
     */
    size_t TakeInputConnections(size_t count);

    /**
     * Assign the next |count| output connections and return the index at which these connections begin.
     */
    size_t TakeOutputConnections(size_t count);

    /**
     * Set the input and output connection indices assigned to each neuron into the neurons themselves.
    */
    void FixNeuronConnectionIndices();

    /**
     * Allocate storage for all the connections this topology will require.
     */
    void AllocateConnections();

    /**
     * Indicates if the connections for the topology have been allocated.
     */
    bool AreConnectionsAllocated() const;

    void ConnectFully();
    void ConnectLayerToNeuron(size_t from_neuron_index, size_t from_neuron_count, size_t to_neuron_index);
    void ConnectLayers(size_t from_neuron_index, size_t from_neuron_count, size_t to_neuron_index, size_t to_neuron_count);
    void ConnectBiasNeuron(size_t bias_neuron_index, size_t to_neuron_index, size_t to_neuron_count);
    void ConnectNeurons(size_t from_neuron_index, size_t to_neuron_index);

private:
    std::vector<Layer> hidden_layers_;
    std::vector<InputConnection> input_connections_;
    size_t input_connection_index_ = 0;
    std::vector<OutputConnection> output_connections_;
    size_t output_connection_index_ = 0;

    bool enable_shortcut_connections_ = false;
    bool is_constructed_ = false;
    bool is_allocated_ = false;
};

} // namespace panann

#endif  // MULTILAYERTOPOLOGY_H__
