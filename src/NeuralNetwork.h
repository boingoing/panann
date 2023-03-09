//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#pragma once

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
        MeanAbsoluteError
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
    NeuralNetwork();
    virtual ~NeuralNetwork();

    /**
     * Set the number of neurons in the input layer.
     * This count may not be changed once the network topology has been constructed.
     */
    void SetInputNeuronCount(size_t inputNeuronCount);
    size_t GetInputNeuronCount();
    size_t GetInputNeuronStartIndex();

    /**
     * Set the number of neurons in the output layer.
     * This count may not be changed once the network topology has been constructed.
     */
    void SetOutputNeuronCount(size_t outputNeuronCount);
    size_t GetOutputNeuronCount();
    size_t GetOutputNeuronStartIndex();

    /**
     * Append a hidden layer to the end of the list of existing hidden layers.<br/>
     * Hidden layers are located after the input layer and before the output layer.<br/>
     * Once added, hidden layers may not be removed.<br/>
     * Hidden layers may not be added after the network has been constructed.
     */
    void AddHiddenLayer(size_t neuronCount);
    size_t GetHiddenNeuronCount();
    size_t GetHiddenNeuronStartIndex();

    /**
     * Set the learning rate parameter used by backprop, batch, and qprop.<br/>
     * Default: 0.7
     */
    void SetLearningRate(double learningRate);
    double GetLearningRate();

    /**
     * Set the momentum parameter used by backprop.<br/>
     * Default: 0.1
     */
    void SetMomentum(double momentum);
    double GetMomentum();

    /**
     * Set the Mu parameter for qprop.<br/>
     * Default: 1.75
     */
    void SetQpropMu(double mu);
    double GetQpropMu();

    /**
     * Set the weight decay parameter for qprop.<br/>
     * Default: -0.0001
     */
    void SetQpropWeightDecay(double weightDecay);
    double GetQpropWeightDecay();

    /**
     * Set the initial weight step value for rprop and sarprop.<br/>
     * This is sometimes called delta_zero.<br/>
     * Default: 0.0125
     */
    void SetRpropWeightStepInitial(double weightStep);
    double GetRpropWeightStepInitial();

    /**
     * Set the minimum weight step value for rprop and sarprop.<br/>
     * This is sometimes called delta_min.<br/>
     * If the weight change becomes 0, learning will stop so we need
     * to use some small value instead of zero.<br/>
     * Default: 0.000001
     */
    void SetRpropWeightStepMin(double weightStep);
    double GetRpropWeightStepMin();

    /**
     * Set the maximum weight step value for rprop and sarprop.<br/>
     * This is sometimes called delta_max.<br/>
     * If the weight change becomes too large, learning will be chaotic
     * so we clamp it to some reasonably large number to ensure smooth
     * training.<br/>
     * Default: 50
     */
    void SetRpropWeightStepMax(double weightStep);
    double GetRpropWeightStepMax();

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
    double GetRpropIncreaseFactor();

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
    double GetRpropDecreaseFactor();

    /**
     * Set the weight decay shift parameter used by sarprop.<br/>
     * This is a constant used as a weight decay.<br/>
     * It is called k1 in some formulations of sarprop.<br/>
     * Default: 0.01
     */
    void SetSarpropWeightDecayShift(double k1);
    double GetSarpropWeightDecayShift();

    /**
     * Set the weight step threshold factor parameter used by sarprop.<br/>
     * It is called k2 in some formulations of sarprop.<br/>
     * Default: 0.1
     */
    void SetSarpropStepThresholdFactor(double k2);
    double GetSarpropStepThresholdFactor();

    /**
     * Set the step shift parameter used by sarprop.<br/>
     * It is called k3 in some formulations of sarprop.<br/>
     * Default: 3
     */
    void SetSarpropStepShift(double k3);
    double GetSarpropStepShift();

    /**
     * Set the temperature parameter used by sarprop.<br/>
     * This is referred to as T in most sarprop formulations.<br/>
     * Default: 0.015
     */
    void SetSarpropTemperature(double t);
    double GetSarpropTemperature();

    /**
     * Set the training algorithm this network will use during training.<br/>
     * Default: ResilientBackpropagation
     * @see TrainingAlgorithmType
     */
    void SetTrainingAlgorithmType(TrainingAlgorithmType type);
    TrainingAlgorithmType GetTrainingAlgorithmType();

    /**
     * Set the cost function we will use to calculate the total network
     * error.<br/>
     * Default: MeanSquareError
     * @see GetError
     * @see ErrorCostFunction
     */
    void SetErrorCostFunction(ErrorCostFunction mode);
    ErrorCostFunction GetErrorCostFunction();

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
    ActivationFunctionType GetHiddenNeuronActivationFunctionType();

    /**
     * Set the default activation function we will use for output layer neurons.<br/>
     * Default: Sigmoid
     */
    void SetOutputNeuronActivationFunctionType(ActivationFunctionType type);
    ActivationFunctionType GetOutputNeuronActivationFunctionType();

    void SetNeuronActivationFunction(size_t neuronIndex, ActivationFunctionType type);

    /**
     * Build the network topology.<br/>
     * After construction, the number of input and output neurons, number of
     * hidden layers, use of shortcut connections, and some other settings may
     * not be modified.
     */
    virtual void Construct();

    /**
     * Reset every weight in the network to a random value between min and max.
     */
    void InitializeWeightsRandom(double min = -1.0, double max = 1.0);

    /**
     * Initialize the weight of each connection via Widrow-Nguyen's algorithm.
     */
    void InitializeWeights(const TrainingData* trainingData);

    /**
     * Use the training algorithm to train the network.<br/>
     * Training follows these steps:<br/>
     *   - For each example in the training data<br/>
     *   - Run the network forward on the example input<br/>
     *   - Calculate the total error by comparing output neuron values against the example output<br/>
     *   - Calculate the partial error contributed by each weight in the network<br/>
     *   - Update all the weights in the network to reduce the total error<br/>
     * Execute the above once for each epoch.<br/>
     * The actual method by which we will update the weights depends on the
     * training algorithm chosen.
     * @param trainingData Examples on which we will train the network.<br/>
     * Note: Shuffles the order of examples in trainingData.
     * @param epochCount The number of epochs we should execute to train the
     * network. One epoch is one full step through all of the training examples.
     * @see SetTrainingAlgorithmType
     * @see TrainingAlgorithmType
     * @see TrainingData
     */
    void Train(TrainingData* trainingData, size_t epochCount);

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
    virtual void RunForward(const std::vector<double>* input);

    /**
     * Compute the error attributed to each neuron in the network.<br/>
     * Begins by calculating the difference between each output neuron value and each
     * element in the output parameter.<br/>
     * Next we pull that neuron error backwards through the network, computing the
     * partial error contributed by each neuron in the network.
     * @param output Must have the same number of elements as this network has output
     * neurons.
     */
    void RunBackward(const std::vector<double>* output);

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
    double GetError(const std::vector<double>* output);

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
    double GetError(const TrainingData* trainingData);

    /**
     * Get a writable vector containing all the weight values for the network.
     */
    std::vector<double>* GetWeights();

    /**
     * Set the weight values for the network.<br/>
     * The weight values will be copied from the provided vector which must contain exactly the number
     * of values for which this network has weights.
     */
    void SetWeights(std::vector<double>* weights);

    /**
     * Writes all of the output neuron values into a vector.<br/>
     * Existing values in the output parameter will be discarded.
     */
    void GetOutput(std::vector<double>* output);

protected:
    NeuralNetwork(const NeuralNetwork&);

    void Allocate();
    void ConnectFully();

    void ConnectLayerToNeuron(size_t fromNeuronIndex, size_t fromNeuronCount, size_t toNeuronIndex);
    void ConnectLayers(size_t fromNeuronIndex, size_t fromNeuronCount, size_t toNeuronIndex, size_t toNeuronCount);
    void ConnectBiasNeuron(size_t biasNeuronIndex, size_t toNeuronIndex, size_t toNeuronCount);
    void ConnectNeurons(size_t fromNeuronIndex, size_t toNeuronIndex);

    void ComputeNeuronValue(size_t neuronIndex);
    void ComputeNeuronValueRange(size_t neuronStartIndex, size_t neuronCount);

    void ComputeNeuronError(size_t neuronIndex);
    void UpdateSlopes();
    void UpdateWeightsOnline();
    void UpdateWeightsOffline(size_t currentEpoch, size_t stepCount);
    void UpdateWeightsBatchingBackpropagation(size_t stepCount);
    void UpdateWeightsQuickBackpropagation(size_t stepCount);
    void UpdateWeightsResilientBackpropagation();
    void UpdateWeightsSimulatedAnnealingResilientBackpropagation(size_t currentEpoch);

    static double ExecuteActivationFunction(Neuron* neuron);
    static double ExecuteActivationFunctionDerivative(Neuron* neuron);
    static bool IsActivationFunctionSymmetric(ActivationFunctionType activationFunctionType);
    static double ApplyErrorShaping(double value);

    size_t GetBiasNeuronStartIndex();

    void ResetWeightSteps();
    void ResetSlopes();
    void ResetPreviousSlopes();
    void TrainOffline(TrainingData* trainingData, size_t epochCount);
    void TrainOnline(TrainingData* trainingData, size_t epochCount);

    void ResetOutputLayerError();
    void CalculateOutputLayerError(const std::vector<double>* output);

    double GetError();

private:
    size_t input_neuron_count_ = 0;
    size_t output_neuron_count_ = 0;
    size_t hidden_neuron_count_ = 0;

    double _learningRate = 0.7;
    double _momentum = 0.1;
    double _qpropMu = 1.75;
    double _qpropWeightDecay = -0.0001;
    double _rpropWeightStepInitial = 0.0125;
    double _rpropWeightStepMin = 0.000001;
    double _rpropWeightStepMax = 50;
    double _rpropIncreaseFactor = 1.2;
    double _rpropDecreaseFactor = 0.5;
    double _sarpropWeightDecayShift = 0.01;
    double _sarpropStepThresholdFactor = 0.1;
    double _sarpropStepShift = 3;
    double _sarpropTemperature = 0.015;

    double _errorSum = 0;
    double _errorCount = 0;

    std::vector<Neuron> _neurons;
    std::vector<Layer> _hiddenLayers;
    std::vector<InputConnection> _inputConnections;
    std::vector<OutputConnection> _outputConnections;
    std::vector<double> _weights;
    std::vector<double> _previousWeightSteps;
    std::vector<double> _slopes;
    std::vector<double> _previousSlopes;

    ActivationFunctionType _hiddenNeuronActivationFunctionType = ActivationFunctionType::Sigmoid;
    ActivationFunctionType _outputNeuronActivationFunctionType = ActivationFunctionType::Sigmoid;
    ErrorCostFunction _errorCostFunction = ErrorCostFunction::MeanSquareError;
    TrainingAlgorithmType _trainingAlgorithmType = TrainingAlgorithmType::ResilientBackpropagation;

    bool _shouldShapeErrorCurve = true;
    bool _enableShortcutConnections = false;
    bool _isConstructed = false;

    RandomWrapper random_;
};

} // namespace panann
