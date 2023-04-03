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

size_t NeuronContainer::AddHiddenNeurons(size_t count) {
    assert(!AreNeuronsAllocated());

    // Remember what the index was when the new neurons were added. Note: This should technically be a hidden neuron index (not an absolute index into |neurons_|) but since the hidden neurons are located at the beginning of that vector, the two indices would be equivalent. If the layout of hidden neurons ever changes, fix this here too.
    const size_t index = hidden_neuron_count_;
    hidden_neuron_count_ += count;
    return index;
}

void NeuronContainer::AddBiasNeurons(size_t count) {
    assert(!AreNeuronsAllocated());
    bias_neuron_count_ += count;
}

void NeuronContainer::AllocateNeurons() {
    assert(!AreNeuronsAllocated());

    // Total count of neurons is all the input, output, hidden, and bias neurons.
    const size_t neuron_count =
        GetInputNeuronCount() +
        GetOutputNeuronCount() +
        GetHiddenNeuronCount() +
        GetBiasNeuronCount();

    neurons_.resize(neuron_count);
    is_allocated_ = true;
}

bool NeuronContainer::AreNeuronsAllocated() const {
    return is_allocated_;
}

NeuronContainer::Neuron& NeuronContainer::GetNeuron(size_t neuron_index) {
    assert(AreNeuronsAllocated());
    assert(neuron_index < neurons_.size());
    return neurons_[neuron_index];
}

NeuronContainer::Neuron& NeuronContainer::GetInputNeuron(size_t input_neuron_index) {
    assert(input_neuron_index < input_neuron_count_);
    return GetNeuron(GetInputNeuronStartIndex() + input_neuron_index);
}

NeuronContainer::Neuron& NeuronContainer::GetOutputNeuron(size_t output_neuron_index) {
    assert(output_neuron_index < output_neuron_count_);
    return GetNeuron(GetOutputNeuronStartIndex() + output_neuron_index);
}

NeuronContainer::Neuron& NeuronContainer::GetBiasNeuron(size_t bias_neuron_index) {
    assert(bias_neuron_index < bias_neuron_count_);
    return GetNeuron(GetBiasNeuronStartIndex() + bias_neuron_index);
}

NeuronContainer::Neuron& NeuronContainer::GetHiddenNeuron(size_t hidden_neuron_index) {
    assert(hidden_neuron_index < hidden_neuron_count_);
    return GetNeuron(GetHiddenNeuronStartIndex() + hidden_neuron_index);
}

const NeuronContainer::Neuron& NeuronContainer::GetNeuron(size_t neuron_index) const {
    assert(neuron_index < neurons_.size());
    return neurons_[neuron_index];
}

const NeuronContainer::Neuron& NeuronContainer::GetOutputNeuron(size_t output_neuron_index) const {
    assert(output_neuron_index < output_neuron_count_);
    return GetNeuron(GetOutputNeuronStartIndex() + output_neuron_index);
}

}  // namespace panann
