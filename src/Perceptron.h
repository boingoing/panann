//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#ifndef PERCEPTRON_H__
#define PERCEPTRON_H__

#include <vector>

#include "ActivationFunction.h"
#include "MultiLayerNeuralTopology.h"
#include "RandomWrapper.h"

namespace panann {

class TrainingData;

/**
 * Simple feed-forward, multi-layer perceptron.
 */
class Perceptron : public MultiLayerNeuralTopology {
public:
    enum class ErrorCostFunction : uint8_t {
        MeanSquareError = 1,
        MeanAbsoluteError,
    };

    Perceptron() = default;
    Perceptron(const Perceptron&) = delete;
    Perceptron& operator=(const Perceptron&) = delete;
    ~Perceptron() override = default;

    /**
     * Set the cost function we will use to calculate the total network
     * error.<br/>
     * Default: MeanSquareError
     * @see GetError
     * @see ErrorCostFunction
     */
    void SetErrorCostFunction(ErrorCostFunction mode);
    ErrorCostFunction GetErrorCostFunction() const;

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
     * Reset every weight in the network to a random value between min and max.
     */
    void InitializeWeightsRandom(double min = -1.0, double max = 1.0);

    /**
     * Initialize the weight of each connection via Widrow-Nguyen's algorithm.
     */
    void InitializeWeights(const TrainingData& training_data);

    /**
     * Run the network forward on a set of inputs.<br/>
     * The values computed by running the network will be stored in the output neurons.<br/>
     * Works by assigning the values in the input parameter into each input neuron
     * and pulling those values through the hidden layers, calculating the output of
     * each neuron as the result of executing the activation function on the sum of the
     * incoming values multiplied by their connection weights.
     * @param input Must have the same number of elements as this network has input
     * neurons.
     */
    virtual void RunForward(const std::vector<double>& input);

    /**
     * Compute the error attributed to each neuron in the network.<br/>
     * Begins by calculating the difference between each output neuron value and each
     * element in the output parameter.<br/>
     * Next we pull that neuron error backwards through the network, computing the
     * partial error contributed by each neuron in the network.
     * @param output Must have the same number of elements as this network has output
     * neurons.
     */
    void RunBackward(const std::vector<double>& output);

    /**
     * Get the total network error by calculating the average of the difference between each
     * output neuron value and each value in the output parameter.<br/>
     * The difference between the expected value and the output neuron value will be
     * modified according to the error cost function.
     * @param output Expected output values. Difference between this and the value of
     * each output neuron is our total network error. The number of values in output must be
     * the same as the number of output neurons in this network.
     * @see ErrorCostFunction
     * @see SetErrorCostFunction
     */
    double GetError(const std::vector<double>& output);

    /**
     * Get the total network error against a set of examples.<br/>
     * For each example, we run the network forward on the input and compute the difference
     * between the output and the values stored in each output neuron.<br/>
     * The total network error is computed as the average of all of those computed differences.<br/>
     * The difference between the expected value and the output neuron value will be
     * modified according to the error cost function.
     * @see ErrorCostFunction
     * @see SetErrorCostFunction
     */
    double GetError(const TrainingData& training_data);

    /**
     * Get a writable vector containing all the weight values for the network.
     */
    std::vector<double>& GetWeights();

    /**
     * Set the weight values for the network based on an input vector |weights|.<br/>
     * The weight values will be copied from the provided vector which must contain exactly the number
     * of values for which this network has weights.
     */
    void SetWeights(const std::vector<double>& weights);

    /**
     * Writes all of the output neuron values into |output|.<br/>
     * Existing values in |output| will be discarded.
     */
    void GetOutput(std::vector<double>* output) const;

    /**
     * Build the neural network.<br/>
     * After construction, the network topology may not be modified.
     */
    void Construct();

    /**
     * Returns true if the network has been constructed and false otherwise.<br/>
     * Note: Once constructed, the network topology is fixed and cannot be changed.
     */
    bool IsConstructed() const;

    /**
     * Enable to perform shaping of the error curve.
     * Default: Disabled
     */
    void EnableErrorShaping();
    void DisableErrorShaping();

protected:
    void AllocateWeights();
    bool AreWeightsAllocated() const;

    /**
     * Set the initial value, activation function, etc for all neurons in the network.
    */
    void InitializeNeurons();

    void ComputeNeuronValue(size_t neuron_index);
    void ComputeNeuronValueRange(size_t neuron_start_index, size_t neuron_count);
    void ComputeNeuronError(size_t neuron_index);

    void ResetOutputLayerError();
    void CalculateOutputLayerError(const std::vector<double>& output);

    double GetError() const;

    RandomWrapper& GetRandom();

    size_t GetWeightCount() const;
    double& GetWeight(size_t index);

private:
    std::vector<double> weights_;
    RandomWrapper random_;

    double error_sum_ = 0;
    double error_count_ = 0;

    ErrorCostFunction error_cost_function_ = ErrorCostFunction::MeanSquareError;
    ActivationFunctionType hidden_neuron_activation_function_type_ = ActivationFunctionType::Sigmoid;
    ActivationFunctionType output_neuron_activation_function_type_ = ActivationFunctionType::Sigmoid;

    bool should_shape_error_curve_ = false;
    bool is_constructed_ = false;
    bool is_allocated_ = false;
};

} // namespace panann

#endif  // PERCEPTRON_H__
