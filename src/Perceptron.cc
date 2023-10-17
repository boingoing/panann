//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for
// full license information.
//-------------------------------------------------------------------------------------------------------

#include "Perceptron.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "ActivationFunction.h"
#include "TrainingData.h"

namespace {

using ActivationFunctionType = panann::ActivationFunctionType;
using panann::ActivationFunction;

bool IsActivationFunctionSymmetric(ActivationFunctionType type) {
  return type >= ActivationFunctionType::FirstSymmetric;
}

double ExecuteActivationFunction(ActivationFunctionType type, double value) {
  switch (type) {
    case ActivationFunctionType::Sigmoid:
      return ActivationFunction::ExecuteSigmoid(value);
    case ActivationFunctionType::SigmoidSymmetric:
      return ActivationFunction::ExecuteSigmoidSymmetric(value);
    case ActivationFunctionType::Gaussian:
      return ActivationFunction::ExecuteGaussian(value);
    case ActivationFunctionType::GaussianSymmetric:
      return ActivationFunction::ExecuteGaussianSymmetric(value);
    case ActivationFunctionType::Cosine:
      return ActivationFunction::ExecuteCosine(value);
    case ActivationFunctionType::CosineSymmetric:
      return ActivationFunction::ExecuteCosineSymmetric(value);
    case ActivationFunctionType::Sine:
      return ActivationFunction::ExecuteSine(value);
    case ActivationFunctionType::SineSymmetric:
      return ActivationFunction::ExecuteSineSymmetric(value);
    case ActivationFunctionType::Elliot:
      return ActivationFunction::ExecuteElliot(value);
    case ActivationFunctionType::ElliotSymmetric:
      return ActivationFunction::ExecuteElliotSymmetric(value);
    case ActivationFunctionType::Linear:
      return ActivationFunction::ExecuteLinear(value);
    case ActivationFunctionType::Threshold:
      return ActivationFunction::ExecuteThreshold(value);
    case ActivationFunctionType::ThresholdSymmetric:
      return ActivationFunction::ExecuteThresholdSymmetric(value);
    default:
      assert(false);
  }

  return 0;
}

double ExecuteActivationFunctionDerivative(ActivationFunctionType type,
                                           double value, double field) {
  switch (type) {
    case ActivationFunctionType::Sigmoid:
      return ActivationFunction::ExecuteDerivativeSigmoid(value);
    case ActivationFunctionType::SigmoidSymmetric:
      return ActivationFunction::ExecuteDerivativeSigmoidSymmetric(value);
    case ActivationFunctionType::Gaussian:
      return ActivationFunction::ExecuteDerivativeGaussian(value, field);
    case ActivationFunctionType::GaussianSymmetric:
      return ActivationFunction::ExecuteDerivativeGaussianSymmetric(value,
                                                                    field);
    case ActivationFunctionType::Cosine:
      return ActivationFunction::ExecuteDerivativeCosine(field);
    case ActivationFunctionType::CosineSymmetric:
      return ActivationFunction::ExecuteDerivativeCosineSymmetric(field);
    case ActivationFunctionType::Sine:
      return ActivationFunction::ExecuteDerivativeSine(field);
    case ActivationFunctionType::SineSymmetric:
      return ActivationFunction::ExecuteDerivativeSineSymmetric(field);
    case ActivationFunctionType::Elliot:
      return ActivationFunction::ExecuteDerivativeElliot(field);
    case ActivationFunctionType::ElliotSymmetric:
      return ActivationFunction::ExecuteDerivativeElliotSymmetric(field);
    case ActivationFunctionType::Linear:
      return ActivationFunction::ExecuteDerivativeLinear(value);
    default:
      assert(false);
  }

  return 0;
}

double ApplyErrorShaping(double value) {
  // TODO(boingoing): Should this be replaced by?
  // return tanh(value);

  static constexpr double tanh_value_limit = 0.9999999;
  static constexpr double tanh_output_clamp = 17;
  if (value < -tanh_value_limit) {
    return -tanh_output_clamp;
  }
  if (value > tanh_value_limit) {
    return tanh_output_clamp;
  }
  return log((1.0 + value) / (1.0 - value));
}

}  // namespace

namespace panann {

void Perceptron::SetErrorCostFunction(ErrorCostFunction mode) {
  error_cost_function_ = mode;
}

Perceptron::ErrorCostFunction Perceptron::GetErrorCostFunction() const {
  return error_cost_function_;
}

void Perceptron::ComputeNeuronValueRange(size_t neuron_start_index,
                                         size_t neuron_count) {
  for (size_t i = 0; i < neuron_count; i++) {
    ComputeNeuronValue(neuron_start_index + i);
  }
}

void Perceptron::ComputeNeuronValue(size_t neuron_index) {
  auto& neuron = GetNeuron(neuron_index);
  neuron.field = 0.0;

  // Sum incoming values.
  for (size_t i = 0; i < neuron.input_connection_count; i++) {
    const size_t input_connection_index =
        neuron.input_connection_start_index + i;
    const auto& connection = GetInputConnection(input_connection_index);
    assert(connection.to_neuron_index == neuron_index);
    const auto& from_neuron = GetNeuron(connection.from_neuron_index);

    neuron.field += from_neuron.value * weights_[input_connection_index];
  }

  neuron.value =
      ExecuteActivationFunction(neuron.activation_function_type, neuron.field);
}

void Perceptron::ComputeNeuronError(size_t neuron_index) {
  auto& neuron = GetNeuron(neuron_index);
  double sum = 0.0;

  // Sum outgoing errors.
  for (size_t i = 0; i < neuron.output_connection_count; i++) {
    const size_t output_connection_index =
        neuron.output_connection_start_index + i;
    const auto& output_connection =
        GetOutputConnection(output_connection_index);
    const auto& input_connection =
        GetInputConnection(output_connection.input_connection_index);
    const auto& to_neuron = GetNeuron(input_connection.to_neuron_index);

    sum += weights_[output_connection.input_connection_index] * to_neuron.error;
  }

  neuron.error =
      ExecuteActivationFunctionDerivative(neuron.activation_function_type,
                                          neuron.value, neuron.field) *
      sum;
}

void Perceptron::RunForward(const std::vector<double>& input) {
  assert(IsConstructed());
  assert(input.size() == GetInputNeuronCount());

  // Feed each input into the corresponding input neuron.
  for (size_t i = 0; i < GetInputNeuronCount(); i++) {
    auto& neuron = GetInputNeuron(i);
    neuron.value = input[i];
  }

  // Pull the values from the input layer through the hidden layer neurons.
  ComputeNeuronValueRange(GetHiddenNeuronStartIndex(), GetHiddenNeuronCount());

  // Pull values into the output layer.
  ComputeNeuronValueRange(GetOutputNeuronStartIndex(), GetOutputNeuronCount());
}

void Perceptron::RunBackward(const std::vector<double>& output) {
  assert(IsConstructed());
  assert(output.size() == GetOutputNeuronCount());

  ResetOutputLayerError();
  CalculateOutputLayerError(output);

  // Calculate error at each hidden layer neuron.
  const size_t hiddenNeuronStartIndex = GetHiddenNeuronStartIndex();
  for (size_t i = 0; i < GetHiddenNeuronCount(); i++) {
    ComputeNeuronError(hiddenNeuronStartIndex +
                       (GetHiddenNeuronCount() - 1 - i));
  }
}

void Perceptron::InitializeWeightsRandom(double min, double max) {
  assert(IsConstructed());

  for (auto& weight : weights_) {
    weight = random_.RandomFloat(min, max);
  }
}

void Perceptron::InitializeWeights(const TrainingData& training_data) {
  assert(IsConstructed());
  assert(!training_data.empty());

  double min_input = std::numeric_limits<double>::max();
  double max_input = std::numeric_limits<double>::min();

  // Search all examples to find the min/max input values.
  for (const auto& example : training_data) {
    assert(GetInputNeuronCount() == example.input.size());
    const auto minmax =
        std::minmax_element(example.input.begin(), example.input.end());
    min_input = std::min(min_input, *minmax.first);
    max_input = std::max(max_input, *minmax.second);
  }

  constexpr double neuron_percentage = 0.7;
  const double factor = pow(neuron_percentage * GetHiddenNeuronCount(),
                            1.0 / GetHiddenNeuronCount()) /
                        (max_input - min_input);
  InitializeWeightsRandom(-factor, factor);
}

double Perceptron::GetError() const {
  assert(error_count_ != 0);
  return error_sum_ / error_count_;
}

double Perceptron::GetError(const std::vector<double>& output) {
  ResetOutputLayerError();
  CalculateOutputLayerError(output);
  return GetError();
}

double Perceptron::GetError(const TrainingData& training_data) {
  ResetOutputLayerError();

  for (const auto& example : training_data) {
    RunForward(example.input);
    CalculateOutputLayerError(example.output);
  }

  return GetError();
}

void Perceptron::ResetOutputLayerError() {
  error_count_ = 0;
  error_sum_ = 0.0;
}

void Perceptron::CalculateOutputLayerError(const std::vector<double>& output) {
  // Calculate error at each output neuron.
  const size_t output_neuron_start_index = GetOutputNeuronStartIndex();
  for (size_t i = 0; i < GetOutputNeuronCount(); i++) {
    auto& neuron = GetNeuron(output_neuron_start_index + i);
    double delta = neuron.value - output[i];

    switch (error_cost_function_) {
      case ErrorCostFunction::MeanSquareError:
        error_sum_ += (delta * delta) / 2;
        break;
      case ErrorCostFunction::MeanAbsoluteError:
        error_sum_ += std::fabs(delta);
        break;
    }
    if (should_shape_error_curve_) {
      delta = ApplyErrorShaping(delta);
    }
    /*
    if (IsActivationFunctionSymmetric(neuron.activation_function_type)) {
        delta /= 2;
    }
    */

    error_count_++;
    neuron.error =
        ExecuteActivationFunctionDerivative(neuron.activation_function_type,
                                            neuron.value, neuron.field) *
        delta;
  }
}

std::vector<double>& Perceptron::GetWeights() { return weights_; }

void Perceptron::SetWeights(const std::vector<double>& weights) {
  assert(weights_.size() == weights.size());
  weights_.assign(weights.cbegin(), weights.cend());
}

void Perceptron::GetOutput(std::vector<double>& output) const {
  output.resize(GetOutputNeuronCount());
  for (size_t i = 0; i < GetOutputNeuronCount(); i++) {
    const auto& neuron = GetOutputNeuron(i);
    output[i] = neuron.value;
  }
}

void Perceptron::InitializeHiddenNeurons() {
  // Set the default activation function for hidden and output neurons.
  for (size_t i = 0; i < GetHiddenNeuronCount(); i++) {
    auto& neuron = GetHiddenNeuron(i);
    neuron.activation_function_type = hidden_neuron_activation_function_type_;
  }
}

void Perceptron::InitializeNeurons() {
  // Bias neurons have a fixed value of 1.0 which we set here.
  for (size_t i = 0; i < GetBiasNeuronCount(); i++) {
    auto& neuron = GetBiasNeuron(i);
    neuron.value = 1;
  }
  for (size_t i = 0; i < GetOutputNeuronCount(); i++) {
    auto& neuron = GetOutputNeuron(i);
    neuron.activation_function_type = output_neuron_activation_function_type_;
  }
  InitializeHiddenNeurons();
}

void Perceptron::SetHiddenNeuronActivationFunctionType(
    ActivationFunctionType type) {
  assert(!IsConstructed());
  hidden_neuron_activation_function_type_ = type;
}

ActivationFunctionType Perceptron::GetHiddenNeuronActivationFunctionType()
    const {
  return hidden_neuron_activation_function_type_;
}

void Perceptron::SetOutputNeuronActivationFunctionType(
    ActivationFunctionType type) {
  assert(!IsConstructed());
  output_neuron_activation_function_type_ = type;
}

ActivationFunctionType Perceptron::GetOutputNeuronActivationFunctionType()
    const {
  return output_neuron_activation_function_type_;
}

void Perceptron::Construct() {
  assert(!IsTopologyConstructed());
  assert(!IsConstructed());
  // Do not support networks with no input neurons, no output neurons, or no
  // hidden neurons.
  assert(GetInputNeuronCount() > 0);
  assert(GetOutputNeuronCount() > 0);
  assert(GetHiddenNeuronCount() > 0);

  AllocateNeurons();
  ConstructTopology();
  InitializeNeurons();
  AllocateWeights();

  is_constructed_ = true;
}

bool Perceptron::IsConstructed() const { return is_constructed_; }

void Perceptron::AllocateWeights() {
  assert(!AreWeightsAllocated());

  weights_.resize(GetInputConnectionCount());

  is_allocated_ = true;
}

bool Perceptron::AreWeightsAllocated() const { return is_allocated_; }

RandomWrapper& Perceptron::GetRandom() { return random_; }

size_t Perceptron::GetWeightCount() const {
  assert(AreWeightsAllocated());
  return weights_.size();
}

double& Perceptron::GetWeight(size_t index) {
  assert(AreWeightsAllocated());
  assert(index < weights_.size());
  return weights_[index];
}

void Perceptron::EnableErrorShaping() { should_shape_error_curve_ = true; }

void Perceptron::DisableErrorShaping() { should_shape_error_curve_ = false; }

}  // namespace panann
