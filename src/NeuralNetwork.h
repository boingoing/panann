//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panga contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#pragma once

#include <vector>

#include "RandomWrapper.h"

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
        Backpropagation = 1,
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
         */
        ResilientBackpropagation,
        SimulatedAnnealingResilientBackpropagation
    };

protected:
    struct Neuron {
        size_t _inputConnectionStartIndex;
        size_t _inputConnectionCount;
        size_t _outputConnectionStartIndex;
        size_t _outputConnectionCount;
        double _field;
        double _value;
        double _error;
        ActivationFunctionType _activationFunctionType;
    };

    struct Layer {
        size_t _neuronStartIndex;
        size_t _neuronCount;
    };

    struct InputConnection {
        size_t _fromNeuronIndex;
        size_t _toNeuronIndex;
    };

    struct OutputConnection {
        size_t _inputConnectionIndex;
    };

    size_t _inputNeuronCount;
    size_t _outputNeuronCount;
    size_t _hiddenNeuronCount;

    double _learningRate;
    double _momentum;
    double _qpropMu;
    double _qpropWeightDecay;
    double _rpropWeightStepInitial;
    double _rpropWeightStepMin;
    double _rpropWeightStepMax;
    double _rpropIncreaseFactor;
    double _rpropDecreaseFactor;
    double _sarpropWeightDecayShift;
    double _sarpropStepThresholdFactor;
    double _sarpropStepShift;
    double _sarpropTemperature;

    double _errorSum;
    double _errorCount;

    std::vector<Neuron> _neurons;
    std::vector<Layer> _hiddenLayers;
    std::vector<InputConnection> _inputConnections;
    std::vector<OutputConnection> _outputConnections;
    std::vector<double> _weights;
    std::vector<double> _previousWeightSteps;
    std::vector<double> _slopes;
    std::vector<double> _previousSlopes;

    ActivationFunctionType _defaultActivationFunction;
    ErrorCostFunction _errorCostFunction;
    TrainingAlgorithmType _trainingAlgorithmType;

    bool _shouldShapeErrorCurve;
    bool _enableShortcutConnections;
    bool _isConstructed;

    RandomWrapper _randomWrapper;

public:
    NeuralNetwork();

    /**
     * Set the number of neurons in the input layer.
     */
    void SetInputNeuronCount(size_t inputNeuronCount);
    size_t GetInputNeuronCount();

    /**
     * Set the number of neurons in the output layer.
     */
    void SetOutputNeuronCount(size_t outputNeuronCount);
    size_t GetOutputNeuronCount();

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
    double SetRpropWeightStepMax();

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

    void SetTrainingAlgorithmType(TrainingAlgorithmType type);
    TrainingAlgorithmType GetTrainingAlgorithmType();

    void AddHiddenLayer(size_t neuronCount);
    void Construct();

    void InitializeWeightsRandom(double min = -1.0, double max = 1.0);

    /**
     * Note: Shuffles the examples in trainingData.
     */
    void Train(TrainingData* trainingData, size_t epochCount);
    void RunForward(const std::vector<double>* input);
    void RunBackward(const std::vector<double>* output);

    double GetError();
    double GetError(const std::vector<double>* output);
    double GetError(const TrainingData* trainingData);

protected:
    NeuralNetwork(const NeuralNetwork&);

    void Allocate();
    void ConnectFully();

    void ConnectLayerToNeuron(size_t fromNeuronIndex, size_t fromNeuronCount, size_t toNeuronIndex);
    void ConnectLayers(size_t fromNeuronIndex, size_t fromNeuronCount, size_t toNeuronIndex, size_t toNeuronCount);
    void ConnectBiasNeuron(size_t biasNeuronIndex, size_t toNeuronIndex, size_t toNeuronCount);
    void ConnectNeurons(size_t fromNeuronIndex, size_t toNeuronIndex);

    void ComputeNeuronValue(size_t neuronIndex);
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

    size_t GetOutputNeuronStartIndex();
    size_t GetInputNeuronStartIndex();
    size_t GetHiddenNeuronStartIndex();
    size_t GetBiasNeuronStartIndex();

    void ResetWeightSteps();
    void ResetSlopes();
    void ResetPreviousSlopes();
    void TrainOffline(TrainingData* trainingData, size_t epochCount);
    void TrainOnline(TrainingData* trainingData, size_t epochCount);

    void ResetOutputLayerError();
    void CalculateOutputLayerError(const std::vector<double>* output);
};
