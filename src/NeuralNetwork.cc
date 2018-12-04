#include <cassert>

#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(size_t inputNeuronCount, size_t outputNeuronCount) :
    _inputNeuronCount(inputNeuronCount),
    _outputNeuronCount(outputNeuronCount),
    _hiddenNeuronCount(0),
    _shouldShapeErrorCurve(true),
    _enableShortcutConnections(true) {
}

size_t NeuralNetwork::GetInputNeuronCount() {
    return this->_inputNeuronCount;
}

size_t NeuralNetwork::GetOutputNeuronCount() {
    return this->_outputNeuronCount;
}

void NeuralNetwork::AddHiddenLayer(size_t neuronCount) {
    Layer layer;

    if (this->_hiddenLayers.empty()) {
        layer._neuronStartIndex = this->_inputNeuronCount + 1 + this->_outputNeuronCount;
    } else {
        const Layer& lastLayer = this->_hiddenLayers.back();
        layer._neuronStartIndex = lastLayer._neuronStartIndex + lastLayer._neuronCount + 1;
    }

    layer._neuronCount = neuronCount;
    this->_hiddenLayers.push_back(layer);

    this->_hiddenNeuronCount += neuronCount;
}

void NeuralNetwork::AllocateConnections() {
    // Do not support networks with no hidden layers.
    assert(!this->_hiddenLayers.empty());

    // Total count of neurons is all the input, output, and hidden neurons.
    // The input layer and each hidden layer also contribute one bias neuron.
    size_t neuronCount =
        this->_inputNeuronCount + 1 +
        this->_outputNeuronCount +
        this->_hiddenNeuronCount + this->_hiddenLayers.size();

    this->_neurons.resize(neuronCount);

    size_t neuronIndex = this->_inputNeuronCount + 1 + this->_outputNeuronCount;
    size_t inputConnectionIndex = 0;
    size_t outputConnectionIndex = 0;
    size_t inputConnectionCount = 0;
    size_t outputConnectionCount = 0;
    size_t currentLayerInputConnectionCount = 0;
    size_t currentLayerOutputConnectionCount = 0;

    // Calculate the connections outgoing from the input layer.
    if (this->_enableShortcutConnections) {
        // The input layer connects to all hidden layers and the output layer.
        currentLayerOutputConnectionCount = this->_hiddenNeuronCount + this->_outputNeuronCount;
    } else {
        // The input layer connects only to the first hidden layer.
        currentLayerOutputConnectionCount = this->_hiddenLayers.front()._neuronCount;
    }

    outputConnectionCount += currentLayerOutputConnectionCount * this->_inputNeuronCount;

    for (size_t i = 0; i < this->_inputNeuronCount; i++) {
        Neuron& neuron = this->_neurons.at(i);
        neuron._outputConnectionStartIndex = outputConnectionIndex;

        outputConnectionIndex += currentLayerOutputConnectionCount;
    }

    Neuron& biasNeuron = this->_neurons.at(this->_inputNeuronCount);
    biasNeuron._outputConnectionStartIndex = outputConnectionIndex;
    outputConnectionIndex = this->_hiddenLayers.front()._neuronCount;

    // Calculate the connections incoming to the output layer.
    if (this->_enableShortcutConnections) {
        // All input and hidden neurons are connected to each output neuron.
        currentLayerInputConnectionCount = this->_inputNeuronCount + this->_hiddenNeuronCount + 1;
    } else {
        // Output neurons are connected only to the last hidden layer.
        currentLayerInputConnectionCount = this->_hiddenLayers.back()._neuronCount + 1;
    }

    inputConnectionCount += currentLayerInputConnectionCount * this->_outputNeuronCount;

    size_t firstOutputNeuronIndex = this->_inputNeuronCount + 1;
    for (size_t i = 0; i < this->_outputNeuronCount; i++) {
        Neuron& neuron = this->_neurons.at(firstOutputNeuronIndex + i);
        neuron._inputConnectionStartIndex = inputConnectionIndex;
        //neuron._inputConnectionCount = currentLayerInputConnectionCount;

        inputConnectionIndex += currentLayerInputConnectionCount;
    }

    // Calculate the connections to and from all hidden layers.
    for (size_t layerIndex = 0; layerIndex < this->_hiddenLayers.size(); layerIndex++) {
        currentLayerInputConnectionCount = 0;
        currentLayerOutputConnectionCount = 0;

        if (this->_enableShortcutConnections) {
            // All hidden layers connect to the input layer when shortcuts are enabled.
            currentLayerInputConnectionCount += this->_inputNeuronCount + 1;

            // Each neuron in this layer connects to the neurons in all previous hidden layers.
            for (size_t previousLayerIndex = 0; previousLayerIndex < layerIndex; previousLayerIndex++) {
                currentLayerInputConnectionCount += this->_hiddenLayers.at(previousLayerIndex)._neuronCount + 1;
            }

            // All hidden layers connect directly to the output layer when shortcuts are enabled.
            currentLayerOutputConnectionCount += this->_outputNeuronCount;

            // This layer connects to all neurons in subsequent hidden layers.
            for (size_t nextLayerIndex = layerIndex + 1; nextLayerIndex < this->_hiddenLayers.size(); nextLayerIndex++) {
                currentLayerOutputConnectionCount += this->_hiddenLayers.at(nextLayerIndex)._neuronCount;
            }
        } else {
            if (layerIndex == 0) {
                // First hidden layer connects to the input layer.
                currentLayerInputConnectionCount += this->_inputNeuronCount + 1;
            } else {
                // This hidden layer connects directly the previous one.
                currentLayerInputConnectionCount += this->_hiddenLayers.at(layerIndex - 1)._neuronCount + 1;
            }

            if (layerIndex == this->_hiddenLayers.size() - 1) {
                // Last hidden layer connects to the output layer.
                currentLayerOutputConnectionCount += this->_outputNeuronCount;
            } else {
                assert(layerIndex + 1 < this->_hiddenLayers.size());

                // This hidden layer connects directly to the next one.
                currentLayerOutputConnectionCount += this->_hiddenLayers.at(layerIndex + 1)._neuronCount;
            }
        }

        const Layer& currentLayer = this->_hiddenLayers.at(layerIndex);

        for (size_t i = 0; i < currentLayer._neuronCount; i++) {
            Neuron& neuron = this->_neurons.at(neuronIndex++);
            neuron._inputConnectionStartIndex = inputConnectionIndex;
            //neuron._inputConnectionCount = currentLayerInputConnectionCount;
            neuron._outputConnectionStartIndex = outputConnectionIndex;
            //neuron._outputConnectionCount = currentLayerOutputConnectionCount;

            inputConnectionIndex += currentLayerInputConnectionCount;
            outputConnectionIndex += currentLayerOutputConnectionCount;
        }

        // Bias neurons cannot have shortcut connections.
        size_t biasOutputConnections = 0;
        if (layerIndex == this->_hiddenLayers.size() - 1) {
            // Bias neuron in the last hidden layer connects to the output layer.
            biasOutputConnections = this->_outputNeuronCount;
        } else {
            // Bias neuron in this hidden layer connects to the next hidden layer.
            biasOutputConnections = this->_hiddenLayers.at(layerIndex + 1)._neuronCount;
        }

        // Bias neurons do not have incoming connections.
        Neuron& biasNeuron = this->_neurons.at(neuronIndex++);
        biasNeuron._outputConnectionStartIndex = outputConnectionIndex;
        outputConnectionIndex += biasOutputConnections;

        inputConnectionCount += currentLayer._neuronCount * currentLayerInputConnectionCount;
        outputConnectionCount += currentLayer._neuronCount * currentLayerOutputConnectionCount + biasOutputConnections;
    }

    this->_inputConnections.resize(inputConnectionCount);
    this->_outputConnections.resize(outputConnectionCount);
    this->_weights.resize(inputConnectionCount);
}

void NeuralNetwork::ConnectNeurons(size_t fromNeuronIndex, size_t toNeuronIndex) {
    Neuron& fromNeuron = this->_neurons.at(fromNeuronIndex);
    Neuron& toNeuron = this->_neurons.at(toNeuronIndex);

    size_t inputConnectionIndex = toNeuron._inputConnectionStartIndex + toNeuron._inputConnectionCount;
    size_t weightIndex = inputConnectionIndex;
    Connection& inputConnection = this->_inputConnections.at(inputConnectionIndex);
    inputConnection._neuron = fromNeuronIndex;
    inputConnection._weightIndex = weightIndex;
    toNeuron._inputConnectionCount++;

    size_t outputConnectionIndex = fromNeuron._outputConnectionStartIndex + fromNeuron._outputConnectionCount;
    Connection& outputConnection = this->_outputConnections.at(outputConnectionIndex);
    outputConnection._neuron = toNeuronIndex;
    outputConnection._weightIndex = weightIndex;
    fromNeuron._outputConnectionCount++;
}

void NeuralNetwork::ConnectLayerToNeuron(size_t fromNeuronIndex, size_t fromNeuronCount, size_t toNeuronIndex) {
    for (size_t i = 0; i < fromNeuronCount; i++) {
        ConnectNeurons(fromNeuronIndex + i, toNeuronIndex);
    }
}

void NeuralNetwork::ConnectLayers(size_t fromNeuronIndex, size_t fromNeuronCount, size_t toNeuronIndex, size_t toNeuronCount) {
    for (size_t i = 0; i < toNeuronCount; i++) {
        ConnectLayerToNeuron(fromNeuronIndex, fromNeuronCount, toNeuronIndex + i);
    }
}

void NeuralNetwork::ConnectBiasNeuron(size_t biasNeuronIndex, size_t toNeuronIndex, size_t toNeuronCount) {
    ConnectLayers(biasNeuronIndex, 1, toNeuronIndex, toNeuronCount);
}

void NeuralNetwork::ConnectFully() {
    assert(!this->_hiddenLayers.empty());

    for (size_t layerIndex = 0; layerIndex < this->_hiddenLayers.size(); layerIndex++) {
        const Layer& currentLayer = this->_hiddenLayers.at(layerIndex);

        if (this->_enableShortcutConnections) {
            // Connect to input layer.
            ConnectLayers(0, this->_inputNeuronCount, currentLayer._neuronStartIndex, currentLayer._neuronCount);

            // Connect to all previous hidden layers.
            for (size_t previousLayerIndex = 0; previousLayerIndex < layerIndex; previousLayerIndex++) {
                const Layer& previousLayer = this->_hiddenLayers.at(previousLayerIndex);
                ConnectLayers(previousLayer._neuronStartIndex,
                              previousLayer._neuronCount,
                              currentLayer._neuronStartIndex,
                              currentLayer._neuronCount);
            }

            // Connect to bias neuron.
            if (layerIndex == 0) {
                // Neurons in the first hidden layer connect to the bias neuron in the input layer.
                ConnectBiasNeuron(this->_inputNeuronCount, currentLayer._neuronStartIndex, currentLayer._neuronCount);
            } else {
                // Neurons in other hidden layers connect to the bias neuron in the previous hidden layer.
                const Layer& previousLayer = this->_hiddenLayers.at(layerIndex - 1);
                ConnectBiasNeuron(previousLayer._neuronStartIndex + previousLayer._neuronCount,
                                     currentLayer._neuronStartIndex,
                                     currentLayer._neuronCount);
            }
        } else {
            if (layerIndex == 0) {
                // Connect to input layer and connect bias neuron.
                ConnectLayers(0, 
                              this->_inputNeuronCount + 1,
                              currentLayer._neuronStartIndex,
                              currentLayer._neuronCount);
            } else {
                // Connect to previous hidden layer and connect bias neuron.
                const Layer& previousLayer = this->_hiddenLayers.at(layerIndex - 1);
                ConnectLayers(previousLayer._neuronStartIndex,
                              previousLayer._neuronCount + 1,
                              currentLayer._neuronStartIndex,
                              currentLayer._neuronCount);
            }
        }
    }

    size_t outputLayerNeuronStartIndex = this->_inputNeuronCount + 1;
    const Layer& previousLayer = this->_hiddenLayers.back();
    if (this->_enableShortcutConnections) {
        // Connect input layer to output layer.
        ConnectLayers(0, this->_inputNeuronCount, outputLayerNeuronStartIndex, this->_outputNeuronCount);

        // Connect all hidden layers to output layer.
        for (size_t i = 0; i < this->_hiddenLayers.size(); i++) {
            const Layer& layer = this->_hiddenLayers.at(i);
            ConnectLayers(layer._neuronStartIndex, layer._neuronCount, outputLayerNeuronStartIndex, this->_outputNeuronCount);
        }

        // Connect output layer to the bias neuron in the last hidden layer.
        ConnectBiasNeuron(previousLayer._neuronStartIndex + previousLayer._neuronCount,
                          outputLayerNeuronStartIndex,
                          this->_outputNeuronCount);
    } else {
        // Connect output layer to the last hidden layer including the bias neuron.
        ConnectLayers(previousLayer._neuronStartIndex,
                      previousLayer._neuronCount + 1,
                      outputLayerNeuronStartIndex,
                      this->_outputNeuronCount);
    }
}

void NeuralNetwork::Construct() {
    this->AllocateConnections();
    this->ConnectFully();
}
