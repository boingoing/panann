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
        Backpropagation = 1,
        BatchingBackpropagation,
        QuickBackpropagation,
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
     * Get the number of neurons in the input layer.
     */
    size_t GetInputNeuronCount();
    void SetInputNeuronCount(size_t inputNeuronCount);

    /**
     * Get the number of neurons in the output layer.
     */
    size_t GetOutputNeuronCount();
    void SetOutputNeuronCount(size_t outputNeuronCount);

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
