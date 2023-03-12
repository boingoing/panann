//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#ifndef NEURALNETWORK_H__
#define NEURALNETWORK_H__

#include <vector>

#include "RandomWrapper.h"

namespace panann {

class TrainingData;

/**
 * Simple feed-forward, multi-layer perceptron.
 */
class NeuralNetwork {
public:
    enum class ActivationFunctionType : uint8_t {
        Linear = 1,
        Sigmoid,
        Gaussian,
        Sine,
        Cosine,
        Elliot,
        Threshold,
        SigmoidSymmetric,
        GaussianSymmetric,
        SineSymmetric,
        CosineSymmetric,
        ElliotSymmetric,
        ThresholdSymmetric,

        // The symmetric activation functions are grouped at the end so we would
        // know that an activation function is symmetric if it's at least the
        // first one in this group.
        FirstSymmetric = SigmoidSymmetric
    };

    enum class ErrorCostFunction : uint8_t {
        MeanSquareError = 1,
        MeanAbsoluteError,
    };

    enum class TrainingAlgorithmType : uint8_t {
        /**
         * Backpropagation with momentum and learning rate parameters.<br/>
         * This is an online learning algorithm and does not perform batching weight
         * updates.
         * @see SetLearningRate
         * @see SetMomentum
         */
        Backpropagation = 0,
        /**
         * Batching backpropagation with learning rate parameter.<br/>
         * This is an offline learning algorithm. It batches together weight updates
         * and modifies the weights at the end of each epoch.
         * @see SetLearningRate
         */
        BatchingBackpropagation,
        /**
         * An implementation of quickprop.<br/>
         * It uses the learning rate, mu, and qprop weight decay parameters.<br/>
         * This is an offline learning algorithm.
         * @see SetLearningRate
         * @see SetQpropMu
         * @see SetQpropWeightDecay
         */
        QuickBackpropagation,
        /**
         * Resilient backprop is a very fast training algorithm designed to move quickly
         * down the error curve when the derivative of the partial error doesn't change.<br/>
         * The implementation here is of iRPROP-. It is an offline learning algorithm.
         * @see SetRpropWeightStepInitial
         * @see SetRpropWeightStepMin
         * @see SetRpropWeightStepMax
         * @see SetRpropIncreaseFactor
         * @see SetRpropDecreaseFactor
         */
        ResilientBackpropagation,
        /**
         * Simulated annealing and weight decay are added to resilient backprop.<br/>
         * Attempts to avoid getting stuck in local minima on the error surface
         * while training the network.<br/>
         * Uses all of the rprop parameters plus some specific to sarprop.
         * @see SetRpropWeightStepInitial
         * @see SetRpropWeightStepMin
         * @see SetRpropWeightStepMax
         * @see SetRpropIncreaseFactor
         * @see SetRpropDecreaseFactor
         * @see SetSarpropWeightDecayShift
         * @see SetSarpropStepThresholdFactor
         * @see SetSarpropStepShift
         * @see SetSarpropTemperature
         */
        SimulatedAnnealingResilientBackpropagation
    };

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
    NeuralNetwork() = default;
    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;
    virtual ~NeuralNetwork() = default;

    /**
     * Set the number of neurons in the input layer.
     * This count may not be changed once the network topology has been constructed.
     */
    void SetInputNeuronCount(size_t input_neuron_count);
    size_t GetInputNeuronCount() const;
    size_t GetInputNeuronStartIndex() const;

    /**
     * Set the number of neurons in the output layer.
     * This count may not be changed once the network topology has been constructed.
     */
    void SetOutputNeuronCount(size_t output_neuron_count);
    size_t GetOutputNeuronCount() const;
    size_t GetOutputNeuronStartIndex() const;

    /**
     * Append a hidden layer to the end of the list of existing hidden layers.<br/>
     * Hidden layers are located after the input layer and before the output layer.<br/>
     * Once added, hidden layers may not be removed.<br/>
     * Hidden layers may not be added after the network has been constructed.
     */
    void AddHiddenLayer(size_t neuron_count);
    size_t GetHiddenNeuronCount() const;
    size_t GetHiddenNeuronStartIndex() const;

    /**
     * Set the learning rate parameter used by backprop, batch, and qprop.<br/>
     * Default: 0.7
     */
    void SetLearningRate(double learning_rate);
    double GetLearningRate() const;

    /**
     * Set the momentum parameter used by backprop.<br/>
     * Default: 0.1
     */
    void SetMomentum(double momentum);
    double GetMomentum() const;

    /**
     * Set the Mu parameter for qprop.<br/>
     * Default: 1.75
     */
    void SetQpropMu(double mu);
    double GetQpropMu() const;

    /**
     * Set the weight decay parameter for qprop.<br/>
     * Default: -0.0001
     */
    void SetQpropWeightDecay(double weight_decay);
    double GetQpropWeightDecay() const;

    /**
     * Set the initial weight step value for rprop and sarprop.<br/>
     * This is sometimes called delta_zero.<br/>
     * Default: 0.0125
     */
    void SetRpropWeightStepInitial(double weight_step);
    double GetRpropWeightStepInitial() const;

    /**
     * Set the minimum weight step value for rprop and sarprop.<br/>
     * This is sometimes called delta_min.<br/>
     * If the weight change becomes 0, learning will stop so we need
     * to use some small value instead of zero.<br/>
     * Default: 0.000001
     */
    void SetRpropWeightStepMin(double weight_step);
    double GetRpropWeightStepMin() const;

    /**
     * Set the maximum weight step value for rprop and sarprop.<br/>
     * This is sometimes called delta_max.<br/>
     * If the weight change becomes too large, learning will be chaotic
     * so we clamp it to some reasonably large number to ensure smooth
     * training.<br/>
     * Default: 50
     */
    void SetRpropWeightStepMax(double weight_step);
    double GetRpropWeightStepMax() const;

    /**
     * Set the factor by which the weight step will be increased each
     * training step when the sign of the partial derivative of the error
     * does not change.<br/>
     * This is sometimes referred to as eta+.<br/>
     * Both rprop and sarprop use this paramter for the same purpose.<br/>
     * A higher value can increase the speed of convergeance but can also
     * make training unstable.<br/>
     * Default: 1.2
     */
    void SetRpropIncreaseFactor(double factor);
    double GetRpropIncreaseFactor() const;

    /**
     * Set the factor by which the weight step will be decreased when the
     * sign of the partial derivative of the error changes.<br/>
     * This is sometimes called eta-.<br/>
     * Both rprop and sarprop use this paramter for the same purpose.<br/>
     * A lower value will make training slower and a higher value can
     * make training unstable.<br/>
     * Default: 0.5
     */
    void SetRpropDecreaseFactor(double factor);
    double GetRpropDecreaseFactor() const;

    /**
     * Set the weight decay shift parameter used by sarprop.<br/>
     * This is a constant used as a weight decay.<br/>
     * It is called k1 in some formulations of sarprop.<br/>
     * Default: 0.01
     */
    void SetSarpropWeightDecayShift(double k1);
    double GetSarpropWeightDecayShift() const;

    /**
     * Set the weight step threshold factor parameter used by sarprop.<br/>
     * It is called k2 in some formulations of sarprop.<br/>
     * Default: 0.1
     */
    void SetSarpropStepThresholdFactor(double k2);
    double GetSarpropStepThresholdFactor() const;

    /**
     * Set the step shift parameter used by sarprop.<br/>
     * It is called k3 in some formulations of sarprop.<br/>
     * Default: 3
     */
    void SetSarpropStepShift(double k3);
    double GetSarpropStepShift() const;

    /**
     * Set the temperature parameter used by sarprop.<br/>
     * This is referred to as T in most sarprop formulations.<br/>
     * Default: 0.015
     */
    void SetSarpropTemperature(double t);
    double GetSarpropTemperature() const;

    /**
     * Set the training algorithm this network will use during training.<br/>
     * Default: ResilientBackpropagation
     * @see TrainingAlgorithmType
     */
    void SetTrainingAlgorithmType(TrainingAlgorithmType type);
    TrainingAlgorithmType GetTrainingAlgorithmType() const;

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
     * Set the activation function which will be used for the neuron at |neuron_index|.
     */
    void SetNeuronActivationFunction(size_t neuron_index, ActivationFunctionType type);

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

    /**
     * Reset every weight in the network to a random value between min and max.
     */
    void InitializeWeightsRandom(double min = -1.0, double max = 1.0);

    /**
     * Initialize the weight of each connection via Widrow-Nguyen's algorithm.
     */
    void InitializeWeights(const TrainingData& training_data);

    /**
     * Use the training algorithm to train the network.<br/>
     * Training follows these steps:<br/>
     *   - For each example in the training data<br/>
     *   - Run the network forward on the example input<br/>
     *   - Calculate the total error by comparing output neuron values against the example output<br/>
     *   - Calculate the partial error contributed by each weight in the network<br/>
     *   - Update all the weights in the network to reduce the total error<br/>
     * Execute the above once for each epoch.<br/>
     * The actual method by which we will update the weights depends on the training algorithm chosen.
     * @param training_data Examples on which we will train the network.<br/>
     * Note: Each epoch, shuffles the order of examples in training_data before performing the training operation.
     * @param epoch_count The number of epochs we should execute to train the network. One epoch is one full step through all of the training examples.
     * @see SetTrainingAlgorithmType
     * @see TrainingAlgorithmType
     * @see TrainingData
     */
    void Train(TrainingData* training_data, size_t epoch_count);

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

protected:
    virtual void Allocate();
    virtual void ConnectFully();

    void ConnectLayerToNeuron(size_t from_neuron_index, size_t from_neuron_count, size_t to_neuron_index);
    void ConnectLayers(size_t from_neuron_index, size_t from_neuron_count, size_t to_neuron_index, size_t to_neuron_count);
    void ConnectBiasNeuron(size_t bias_neuron_index, size_t to_neuron_index, size_t to_neuron_count);
    void ConnectNeurons(size_t from_neuron_index, size_t to_neuron_index);

    void ComputeNeuronValue(size_t neuron_index);
    void ComputeNeuronValueRange(size_t neuron_start_index, size_t neuron_count);

    void ComputeNeuronError(size_t neuron_index);
    void UpdateSlopes();
    void UpdateWeightsOnline();
    void UpdateWeightsOffline(size_t current_epoch, size_t step_count);
    void UpdateWeightsBatchingBackpropagation(size_t step_count);
    void UpdateWeightsQuickBackpropagation(size_t step_count);
    void UpdateWeightsResilientBackpropagation();
    void UpdateWeightsSimulatedAnnealingResilientBackpropagation(size_t current_epoch);

    size_t GetBiasNeuronStartIndex() const;

    void ResetWeightSteps();
    void ResetSlopes();
    void ResetPreviousSlopes();
    void TrainOffline(TrainingData* training_data, size_t epoch_count);
    void TrainOnline(TrainingData* training_data, size_t epoch_count);

    void ResetOutputLayerError();
    void CalculateOutputLayerError(const std::vector<double>& output);

    double GetError() const;

    Neuron& GetNeuron(size_t neuron_index);

private:
  static constexpr double DefaultLearningRate = 0.7;
  static constexpr double DefaultMomentum = 0.1;
  static constexpr double DefaultQpropMu = 1.75;
  static constexpr double DefaultQpropWeightDecay = -0.0001;
  static constexpr double DefaultRpropWeightStepInitial = 0.0125;
  static constexpr double DefaultRpropWeightStepMin = 0.000001;
  static constexpr double DefaultRpropWeightStepMax = 50;
  static constexpr double DefaultRpropIncreaseFactor = 1.2;
  static constexpr double DefaultRpropDecreaseFactor = 0.5;
  static constexpr double DefaultSarpropWeightDecayShift = 0.01;
  static constexpr double DefaultSarpropStepThresholdFactor = 0.1;
  static constexpr double DefaultSarpropStepShift = 3;
  static constexpr double DefaultSarpropTemperature = 0.015;

    std::vector<Neuron> neurons_;
    std::vector<Layer> hidden_layers_;
    std::vector<InputConnection> input_connections_;
    std::vector<OutputConnection> output_connections_;
    std::vector<double> weights_;
    std::vector<double> previous_weight_steps_;
    std::vector<double> slopes_;
    std::vector<double> previous_slopes_;

    RandomWrapper random_;

    size_t input_neuron_count_ = 0;
    size_t output_neuron_count_ = 0;
    size_t hidden_neuron_count_ = 0;

    double error_sum_ = 0;
    double error_count_ = 0;

    double learning_rate_ = DefaultLearningRate;
    double momentum_ = DefaultMomentum;
    double qprop_mu_ = DefaultQpropMu;
    double qprop_weight_decay_ = DefaultQpropWeightDecay;
    double rprop_weight_step_initial_ = DefaultRpropWeightStepInitial;
    double rprop_weight_step_min_ = DefaultRpropWeightStepMin;
    double rprop_weight_step_max_ = DefaultRpropWeightStepMax;
    double rprop_increase_factor_ = DefaultRpropIncreaseFactor;
    double rprop_decrease_factor_ = DefaultRpropDecreaseFactor;
    double sarprop_weight_decay_shift_ = DefaultSarpropWeightDecayShift;
    double sarprop_step_threshold_factor_ = DefaultSarpropStepThresholdFactor;
    double sarprop_step_shift_ = DefaultSarpropStepShift;
    double sarprop_temperature_ = DefaultSarpropTemperature;

    ActivationFunctionType hidden_neuron_activation_function_type_ = ActivationFunctionType::Sigmoid;
    ActivationFunctionType output_neuron_activation_function_type_ = ActivationFunctionType::Sigmoid;
    ErrorCostFunction error_cost_function_ = ErrorCostFunction::MeanSquareError;
    TrainingAlgorithmType training_algorithm_type_ = TrainingAlgorithmType::ResilientBackpropagation;

    bool should_shape_error_curve_ = true;
    bool enable_shortcut_connections_ = false;
    bool is_constructed_ = false;
};

} // namespace panann

#endif  // NEURALNETWORK_H__
