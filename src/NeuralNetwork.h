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
        SigmoidSymmetric,
        GaussianSymmetric,
        Sigmoid,
        Gaussian,
        Sine,
        Cosine,
        SineSymmetric,
        CosineSymmetric,
        Elliot,
        ElliotSymmetric,
        Threshold,
        ThresholdSymmetric
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

    struct Connection {
        /**
         * Which neuron are we connected to?<br/>
         * For an input connection, this is the from neuron.<br/>
         * For an output connection, this is the to neuron.
         */
        size_t _neuron;

        /**
         * Index of the weight for this connection.
         */
        size_t _weightIndex;
    };

    size_t _inputNeuronCount;
    size_t _outputNeuronCount;
    size_t _hiddenNeuronCount;

    std::vector<Neuron> _neurons;
    std::vector<double> _weights;
    std::vector<Connection> _inputConnections;
    std::vector<Connection> _outputConnections;
    std::vector<Layer> _hiddenLayers;

    bool _shouldShapeErrorCurve;
    bool _enableShortcutConnections;

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

    void AddHiddenLayer(size_t neuronCount);
    void Construct();

    void InitializeWeightsRandom(double min = -1.0, double max = 1.0);

    void Train(TrainingData* trainingData, size_t epochCount);
    void RunForward(std::vector<double>* input);

protected:
    NeuralNetwork(const NeuralNetwork&);

    void AllocateConnections();
    void ConnectFully();

    void ConnectLayerToNeuron(size_t fromNeuronIndex, size_t fromNeuronCount, size_t toNeuronIndex);
    void ConnectLayers(size_t fromNeuronIndex, size_t fromNeuronCount, size_t toNeuronIndex, size_t toNeuronCount);
    void ConnectBiasNeuron(size_t biasNeuronIndex, size_t toNeuronIndex, size_t toNeuronCount);
    void ConnectNeurons(size_t fromNeuronIndex, size_t toNeuronIndex);

    void ComputeNeuronValue(size_t neuronIndex);
    double ExecuteActivationFunction(ActivationFunctionType activationFunctionType, double field);

    size_t GetOutputNeuronStartIndex();
    size_t GetInputNeuronStartIndex();
    size_t GetHiddenNeuronStartIndex();
    size_t GetBiasNeuronStartIndex();
};
