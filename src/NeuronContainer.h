//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#ifndef NEURONCONTAINER_H__
#define NEURONCONTAINER_H__

#include <vector>

#include "ActivationFunction.h"

namespace panann {

/**
 * Container for storing and lightly organizing a set of neurons.<br/>
 * Intended mainly to act as a base class for artificial neural networks and facilitate neuron storage for the same.<br/>
 *
   * All of the neurons are physically stored in a vector.<br/>
   * All of the hidden neurons are stored first, starting at index 0 and continuing for GetHiddenNeuronCount() neurons, stored sequentially.<br/>
   * Immediately following the hidden neurons are all of the input, output, and bias neurons.<br/>
   *
   * +--------------------------------------------------+----------------------+
   * |          Index                                   |     Which Neuron     |
   * +--------------------------------------------------+----------------------+
   * | GetHiddenNeuronStartIndex()                      | First Hidden Neuron  |
   * +--------------------------------------------------+----------------------+
   * | GetHiddenNeuronStartIndex() + 1                  | Second Hidden Neuron |
   * +--------------------------------------------------+----------------------+
   * | ...                                              |                      |
   * +--------------------------------------------------+----------------------+
   * | GetHiddenNeuronStartIndex()                      | Last Hidden Neuron   |
   * | + GetHiddenNeuronCount() - 1                     |                      |
   * +--------------------------------------------------+----------------------+
   * | GetInputNeuronStartIndex()                       | First Input Neuron   |
   * +--------------------------------------------------+----------------------+
   * | GetInputNeuronStartIndex() + 1                   | Second Input Neuron  |
   * +--------------------------------------------------+----------------------+
   * | ...                                              |                      |
   * +--------------------------------------------------+----------------------+
   * | GetInputNeuronStartIndex()                       | Last Input Neuron    |
   * | + GetInputNeuronCount() - 1                      |                      |
   * +--------------------------------------------------+----------------------+
   * | GetOutputNeuronStartIndex()                      | First Output Neuron  |
   * +--------------------------------------------------+----------------------+
   * | GetOutputNeuronStartIndex() + 1                  | Second Output Neuron |
   * +--------------------------------------------------+----------------------+
   * | ...                                              |                      |
   * +--------------------------------------------------+----------------------+
   * | GetOutputNeuronStartIndex()                      | Last Output Neuron   |
   * | + GetOutputNeuronCount() - 1                     |                      |
   * +--------------------------------------------------+----------------------+
   * | GetBiasNeuronStartIndex()                        | First Bias Neuron    |
   * +--------------------------------------------------+----------------------+
   * | GetBiasNeuronStartIndex() + 1                    | Second Bias Neuron   |
   * +--------------------------------------------------+----------------------+
   * | ...                                              |                      |
   * +--------------------------------------------------+----------------------+
   * | GetBiasNeuronStartIndex()                        | Last Bias Neuron     |
   * | + GetBiasNeuronCount() - 1                       |                      |
   * +--------------------------------------------------+----------------------+
 */
class NeuronContainer {
protected:
    struct Neuron {
        size_t input_connection_start_index;
        size_t input_connection_count;
        size_t output_connection_start_index;
        size_t output_connection_count;
        double field;
        double value;
        double error;
        ActivationFunctionType activation_function_type;
    };

public:
    NeuronContainer() = default;
    NeuronContainer(const NeuronContainer&) = delete;
    NeuronContainer& operator=(const NeuronContainer&) = delete;
    virtual ~NeuronContainer() = default;

    /**
     * Set the number of input neurons in the container.
     * This count may not be changed once the neurons have been allocated.
     */
    void SetInputNeuronCount(size_t input_neuron_count);

    /**
     * Get the count of input neurons in the container.
     */
    size_t GetInputNeuronCount() const;

    /**
     * Set the number of output neurons in the container.
     * This count may not be changed once the neurons have been allocated.
     */
    void SetOutputNeuronCount(size_t output_neuron_count);

    /**
     * Get the count of output neurons in the container.
     */
    size_t GetOutputNeuronCount() const;

    /**
     * Get the total count of neurons in the container.
     * Note: The neurons must have been allocated.
     */
    size_t GetNeuronCount() const;

    /**
     * Get a read-only view of the neuron at |neuron_index|.
     */
    const Neuron& GetNeuron(size_t neuron_index) const;

protected:
/**
 * Get the index of the first hidden neuron. Following this index, there will be GetHiddenNeuronCount() more hidden neurons.
 * @see GetHiddenNeuronCount
 */
  size_t GetHiddenNeuronStartIndex() const;

/**
 * Get the index of the first input neuron. Following this index, there will be GetInputNeuronCount() more input neurons.
 * @see GetInputNeuronCount
 */
  size_t GetInputNeuronStartIndex() const;

/**
 * Get the index of the first output neuron. Following this index, there will be GetOutputNeuronCount() more output neurons.
 * @see GetOutputNeuronCount
 */
    size_t GetOutputNeuronStartIndex() const;

/**
 * Get the index of the first bias neuron. Following this index, there will be GetBiasNeuronCount() more bias neurons.
 * @see GetBiasNeuronCount
 */
    size_t GetBiasNeuronStartIndex() const;

    /**
     * Get the count of hidden neurons in the container.
     */
    size_t GetHiddenNeuronCount() const;

    /**
     * Get the count of bias neurons in the container.
     */
    size_t GetBiasNeuronCount() const;

    /**
     * Adds |count| hidden neurons to the container.<br/>
     * Note: There isn't a way to remove neurons once they're added.
     * @return Index at which the new neurons begin.
    */
    size_t AddHiddenNeurons(size_t count);

    /**
     * Adds |count| bias neurons to the container.<br/>
     * Note: There isn't a way to remove neurons once they're added.
     */
    void AddBiasNeurons(size_t count);

    /**
     * Perform allocation of the vector of neurons.
     * Note: Do this before attempting to use the neurons.
     */
    void AllocateNeurons();

    /**
     * Returns true if the neurons in the container have been allocated.
     */
    bool AreNeuronsAllocated() const;

    /**
     * Set the activation function which will be used for the neuron at |neuron_index|.
     */
    void SetNeuronActivationFunction(size_t neuron_index, ActivationFunctionType type);

    /**
     * Get a writable view of the neuron at |neuron_index|.
     */
    Neuron& GetNeuron(size_t neuron_index);

    /**
     * Get a writable view of the input neuron at index |input_neuron_index|.
     * Note: |input_neuron_index| is not a global index into |neurons_| but must be in the range [0, GetInputNeuronCount()).
     */
    Neuron& GetInputNeuron(size_t input_neuron_index);

    /**
     * Get a writable view of the output neuron at index |output_neuron_index|.
     * Note: |output_neuron_index| is not a global index into |neurons_| but must be in the range [0, GetOutputNeuronCount()).
     */
    Neuron& GetOutputNeuron(size_t output_neuron_index);

    /**
     * Get a writable view of the bias neuron at index |bias_neuron_index|.
     * Note: |bias_neuron_index| is not a global index into |neurons_| but must be in the range [0, GetBiasNeuronCount()).
     */
    Neuron& GetBiasNeuron(size_t bias_neuron_index);

    /**
     * Get a writable view of the hidden neuron at index |hidden_neuron_index|.
     * Note: |hidden_neuron_index| is not a global index into |neurons_| but must be in the range [0, GetHiddenNeuronCount()).
     */
    Neuron& GetHiddenNeuron(size_t hidden_neuron_index);

    /**
     * Get a read-only view of the output neuron at index |output_neuron_index|.
     * Note: |output_neuron_index| is not a global index into |neurons_| but must be in the range [0, GetOutputNeuronCount()).
     */
    const Neuron& GetOutputNeuron(size_t output_neuron_index) const;

private:
    std::vector<Neuron> neurons_;
    size_t input_neuron_count_ = 0;
    size_t output_neuron_count_ = 0;
    size_t hidden_neuron_count_ = 0;

    // There is always a bias neuron which will be hooked-up to output layer neurons.
    size_t bias_neuron_count_ = 1;
    bool is_allocated_ = false;
};

} // namespace panann

#endif  // NEURONCONTAINER_H__
