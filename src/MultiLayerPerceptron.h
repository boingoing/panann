//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#ifndef MULTILAYERPERCEPTRON_H__
#define MULTILAYERPERCEPTRON_H__

#include <vector>

#include "NeuronContainer.h"

namespace panann {

/**
 * Supports building multi-layer topologies made up of neurons.
 */
class MultiLayerPerceptron : public NeuronContainer {
protected:
    struct Layer {
        size_t neuron_start_index;
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
    MultiLayerPerceptron() = default;
    MultiLayerPerceptron(const MultiLayerPerceptron&) = delete;
    MultiLayerPerceptron& operator=(const MultiLayerPerceptron&) = delete;
    ~MultiLayerPerceptron() override = default;

    /**
     * Append a hidden layer to the end of the list of existing hidden layers.<br/>
     * Hidden layers are located after the input layer and before the output layer.<br/>
     * Once added, hidden layers may not be removed.<br/>
     * Hidden layers may not be added after the network has been constructed.
     */
    void AddHiddenLayer(size_t neuron_count);

    /**
     * Shortcut connections are feed-forward connections between two
     * non-adjacent layers.<br/>
     * Note: Changing this setting after the network has been constructed
     * will have no impact on the network topology.<br/>
     * Default: disabled
     */
    void EnableShortcutConnections();
    void DisableShortcutConnections();

    /**
     * Set the default activation function we will use for hidden layer neurons.<br/>
     * Default: Sigmoid
     */
    void SetHiddenNeuronActivationFunctionType(ActivationFunctionType type);
    ActivationFunctionType GetHiddenNeuronActivationFunctionType() const;

    /**
     * Set the default activation function we will use for output layer neurons.<br/>
     * Default: Sigmoid
     */
    void SetOutputNeuronActivationFunctionType(ActivationFunctionType type);
    ActivationFunctionType GetOutputNeuronActivationFunctionType() const;

    /**
     * Build the network topology.<br/>
     * After construction, the number of input and output neurons, number of
     * hidden layers, use of shortcut connections, and some other settings may
     * not be modified.
     */
    virtual void Construct();

    /**
     * Returns true if the network topology has been constructed and false otherwise.<br/>
     * Note: Once constructed, the network topology is fixed and cannot be changed.
     */
    bool IsConstructed() const;

protected:
    size_t GetHiddenLayerCount() const;
    Layer& GetHiddenLayer(size_t layer_index);
    const Layer& GetHiddenLayer(size_t layer_index) const;

    /**
     * Get the count of all input connections in the network topology.
     */
    size_t GetInputConnectionCount() const;
    size_t GetOutputConnectionCount() const;

    InputConnection& GetInputConnection(size_t index);
    OutputConnection& GetOutputConnection(size_t index);

    size_t TakeInputConnections(size_t count);
    size_t TakeOutputConnections(size_t count);

    /**
     * Set the input and output connection indices assigned to each neuron into the neurons themselves
    */
    void FixNeuronConnectionIndices();

    /**
     * Set the initial value, activation function, etc for all neurons in the topology.
    */
    void InitializeNeurons();

    void AllocateConnections();
    bool AreConnectionsAllocated() const;

    virtual void Allocate();
    virtual void ConnectFully();

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

    ActivationFunctionType hidden_neuron_activation_function_type_ = ActivationFunctionType::Sigmoid;
    ActivationFunctionType output_neuron_activation_function_type_ = ActivationFunctionType::Sigmoid;

    bool enable_shortcut_connections_ = false;
    bool is_constructed_ = false;
    bool is_allocated_ = false;
};

} // namespace panann

#endif  // MULTILAYERPERCEPTRON_H__
