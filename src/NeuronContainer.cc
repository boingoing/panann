//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <cassert>

#include "ActivationFunction.h"
#include "NeuronContainer.h"

namespace panann {

size_t NeuronContainer::GetInputNeuronCount() const {
    return input_neuron_count_;
}

void NeuronContainer::SetInputNeuronCount(size_t input_neuron_count) {
    assert(!AreNeuronsAllocated());
    input_neuron_count_ = input_neuron_count;

    // The input layer hooks-up to a bias neuron.
    // TODO(boingoing): We don't really have any other place where we can initialize the input layer - if we don't set the count, it'll be zero which will assert when constructing - so increment the bias neuron count here. We should do this elsewhere if there become other ways to initialize the input layer.
    AddBiasNeurons(1);
}

size_t NeuronContainer::GetOutputNeuronCount() const {
    return output_neuron_count_;
}

void NeuronContainer::SetOutputNeuronCount(size_t output_neuron_count) {
    assert(!AreNeuronsAllocated());
    output_neuron_count_ = output_neuron_count;
}

size_t NeuronContainer::GetHiddenNeuronStartIndex() const {
    return 0;
}

size_t NeuronContainer::GetInputNeuronStartIndex() const {
    return hidden_neuron_count_;
}

size_t NeuronContainer::GetOutputNeuronStartIndex() const {
    return hidden_neuron_count_ + input_neuron_count_;
}

size_t NeuronContainer::GetBiasNeuronStartIndex() const {
    return hidden_neuron_count_ + input_neuron_count_ + output_neuron_count_;
}

size_t NeuronContainer::GetHiddenNeuronCount() const {
    return hidden_neuron_count_;
}

size_t NeuronContainer::GetBiasNeuronCount() const {
    return bias_neuron_count_;
}

void NeuronContainer::AddHiddenNeurons(size_t count) {
    assert(!AreNeuronsAllocated());
    hidden_neuron_count_ += count;
}

void NeuronContainer::AddBiasNeurons(size_t count) {
    assert(!AreNeuronsAllocated());
    bias_neuron_count_ += count;
}

void NeuronContainer::AllocateNeurons() {
    // Total count of neurons is all the input, output, hidden, and bias neurons.
    const size_t neuron_count =
        GetInputNeuronCount() +
        GetOutputNeuronCount() +
        GetHiddenNeuronCount() +
        GetBiasNeuronCount();

    neurons_.resize(neuron_count);
}

bool NeuronContainer::AreNeuronsAllocated() const {
    return is_allocated_;
}

NeuronContainer::Neuron& NeuronContainer::GetNeuron(size_t neuron_index) {
    assert(neuron_index < neurons_.size());
    return neurons_[neuron_index];
}

}  // namespace panann
