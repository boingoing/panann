//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include "FeedForwardNeuralNetwork.h"
#include "Perceptron.h"
#include "RecurrentNeuralNetwork.h"
#include "TrainingData.h"

using panann::FeedForwardNeuralNetwork;
using panann::Perceptron;
using panann::RecurrentNeuralNetwork;
using panann::TrainingData;

namespace testing {

const char* TrainingAlgorithmNames[] = {
    "Backpropagation",
    "BatchingBackpropagation",
    "QuickBackpropagation",
    "ResilientBackpropagation",
    "SimulatedAnnealingResilientBackpropagation"
};

struct TestNeuron {
    size_t input_connection_start_index;
    size_t input_connection_count;
    size_t output_connection_start_index;
    size_t output_connection_count;
//    FeedForwardNeuralNetwork::TrainingAlgorithmType activation_function_type;
};

struct TestInputConnection {
    size_t from_neuron_index;
    size_t to_neuron_index;
};

// Output connection is just the index of an input connection.
using TestOutputConnection = size_t;

struct TestLayer {
    size_t neuron_start_index;
    size_t neuron_count;
};

struct TestPerceptron {
    std::vector<TestLayer> layers;
    std::vector<TestNeuron> neurons;
    std::vector<TestInputConnection> input_connections;
    std::vector<TestOutputConnection> output_connections;
};

struct PerceptronTestCase {
    bool enable_shortcuts;
    size_t input_neurons;
    size_t output_neurons;
    TestPerceptron perceptron;
};

const PerceptronTestCase feedForwardTestCases[] = {{
    // Enable shortcuts
    false,
    // Input neuron count
    2,
    // Output neuron count
    1,
    // Perceptron - what the network topology should look like.
    {
    // Layers: neuron_idx, neuron_cnt
    {
        {0, 3},
    },
    // Neurons: input_idx, input_cnt, output_idx, output_cnt
    {
        // Hidden neurons
        {4,3,6,1},
        {7,3,7,1},
        {10,3,8,1},
        // Input neurons
        {0,0,0,3},
        {0,0,3,3},
        // Output neurons
        {0,4,0,0},
        // Bias neurons
        {0,0,9,3},
        {0,0,12,1},
    },
    // Input connections: from_idx, to_idx
    {
        {0,5},
        {1,5},
        {2,5},
        {7,5},
        {3,0},
        {4,0},
        {6,0},
        {3,1},
        {4,1},
        {6,1},
        {3,2},
        {4,2},
        {6,2},
    },
    // Output connections: input_idx
    {
        4, 7, 10, 5, 8, 11, 0, 1, 2, 6, 9, 12, 3,
    }
    }
},{
    // Enable shortcuts
    false,
    // Input neuron count
    2,
    // Output neuron count
    1,
    // Perceptron - what the network topology should look like.
    {
    // Layers: neuron_idx, neuron_cnt
    {
        {0, 2},
        {2, 2},
    },
    // Neurons: input_idx, input_cnt, output_idx, output_cnt
    {
        // Hidden neurons
        {3,3,4,2},
        {6,3,6,2},
        {9,3,8,1},
        {12,3,9,1},
        // Input neurons
        {0,0,0,2},
        {0,0,2,2},
        // Output neurons
        {0,3,0,0},
        // Bias neurons
        {0,0,10,2},
        {0,0,12,2},
        {0,0,14,1},
    },
    // Input connections: from_idx, to_idx
    {
        {2,6},
        {3,6},
        {9,6},
        {4,0},
        {5,0},
        {7,0},
        {4,1},
        {5,1},
        {7,1},
        {0,2},
        {1,2},
        {8,2},
        {0,3},
        {1,3},
        {8,3},
    },
    // Output connections: input_idx
    {
        3, 6, 4, 7, 9, 12, 10, 13, 0, 1, 5, 8, 11, 14, 2
    }
    }
}};

void MakeXorTwoBitTrainingData(TrainingData* training_data) {
    training_data->resize(4);
    training_data->at(0).input = { 0.0, 0.0 };
    training_data->at(0).output = { 0.0 };
    training_data->at(1).input = { 1.0, 1.0 };
    training_data->at(1).output = { 0.0 };
    training_data->at(2).input = { 1.0, 0.0 };
    training_data->at(2).output = { 1.0 };
    training_data->at(3).input = { 0.0, 1.0 };
    training_data->at(3).output = { 1.0 };
}

void MakeSineTrainingData(TrainingData* training_data, size_t steps) {
    const size_t sine_sample_count = steps * 2;
    const double step_size = 1.0 / sine_sample_count;
    constexpr double pi = 3.14159265358979323846;
    std::vector<double> sine_data;
    sine_data.resize(sine_sample_count);

    for (size_t i = 0; i < sine_sample_count; i++) {
        sine_data[i] = std::sin(step_size * i * 2 * pi);
    }

    training_data->FromSequentialData(sine_data, steps);
}

void MakeTestNetwork(FeedForwardNeuralNetwork* nn, TrainingData* training_data) {
    constexpr size_t neurons_per_hidden_layer = 5;

    nn->DisableShortcutConnections();
    nn->SetInputNeuronCount(training_data->at(0).input.size());
    nn->SetOutputNeuronCount(training_data->at(0).output.size());
    nn->AddHiddenLayer(neurons_per_hidden_layer);
    nn->AddHiddenLayer(neurons_per_hidden_layer);
    nn->Construct();
}

void MakeTestNetwork(RecurrentNeuralNetwork* rnn, TrainingData* training_data) {
    constexpr size_t cells_per_layer = 5;
    constexpr size_t cell_memory_size = 3;

    rnn->SetInputNeuronCount(training_data->at(0).input.size());
    rnn->SetOutputNeuronCount(training_data->at(0).output.size());
    rnn->SetCellMemorySize(cell_memory_size);
    rnn->AddHiddenLayer(cells_per_layer, {});
    rnn->AddHiddenLayer(cells_per_layer, {});
    rnn->Construct();
}

void Compare(size_t left, size_t right, const char* msg) {
    if (left != right) {
        std::cout << "Fail! Expected: " << left << " Found: " << right << ". " << msg << std::endl;
        exit(-1);
    }
}

void ComparePerceptron(const TestPerceptron& test_perceptron, const Perceptron& perceptron) {
  Compare(test_perceptron.layers.size(), perceptron.GetHiddenLayerCount(), "Hidden layer count");
  for (size_t i = 0; i < test_perceptron.layers.size(); i++) {
    const auto& expected_layer = test_perceptron.layers[i];
    const auto& layer = perceptron.GetHiddenLayer(i);
    Compare(expected_layer.neuron_start_index, layer.neuron_start_index, "Neuron start index in the hidden layer");
    Compare(expected_layer.neuron_count, layer.neuron_count, "Neuron count in the hidden layer");
  }

  Compare(test_perceptron.neurons.size(), perceptron.GetNeuronCount(), "Neuron count");
  for (size_t i = 0; i < test_perceptron.neurons.size(); i++) {
    const auto& expected_neuron = test_perceptron.neurons[i];
    const auto& neuron = perceptron.GetNeuron(i);
    Compare(expected_neuron.input_connection_start_index, neuron.input_connection_start_index, "Neuron input connection start index");
    Compare(expected_neuron.input_connection_count, neuron.input_connection_count, "Neuron input connection count");
    Compare(expected_neuron.output_connection_start_index, neuron.output_connection_start_index, "Neuron output connection start index");
    Compare(expected_neuron.output_connection_count, neuron.output_connection_count, "Neuron output connection count");
  }

  Compare(test_perceptron.input_connections.size(), perceptron.GetInputConnectionCount(), "Input connection count");
  for (size_t i = 0; i < test_perceptron.input_connections.size(); i++) {
    const auto& expected_input_connection = test_perceptron.input_connections[i];
    const auto& input_connection = perceptron.GetInputConnection(i);
    Compare(expected_input_connection.from_neuron_index, input_connection.from_neuron_index, "Input connection from neuron");
    Compare(expected_input_connection.to_neuron_index, input_connection.to_neuron_index, "Input connection from neuron");
  }

  Compare(test_perceptron.output_connections.size(), perceptron.GetOutputConnectionCount(), "Output connection count");
  for (size_t i = 0; i < test_perceptron.output_connections.size(); i++) {
    const auto& expected_output_connection = test_perceptron.output_connections[i];
    const auto& output_connection = perceptron.GetOutputConnection(i);
    Compare(expected_output_connection, output_connection.input_connection_index, "Output connection input connection index");
  }
}

void MakeTestNetwork(FeedForwardNeuralNetwork* nn, const PerceptronTestCase& test) {
    if (test.enable_shortcuts) {
        nn->EnableShortcutConnections();
    }
    nn->SetInputNeuronCount(test.input_neurons);
    nn->SetOutputNeuronCount(test.output_neurons);
    for (const auto& layer : test.perceptron.layers) {
        nn->AddHiddenLayer(layer.neuron_count);
    }
    nn->Construct();
}

void TestConstruct(const PerceptronTestCase& test) {
    FeedForwardNeuralNetwork nn;
    MakeTestNetwork(&nn, test);
    ComparePerceptron(test.perceptron, nn);
}

void TrainAndTestNetwork(FeedForwardNeuralNetwork* nn, TrainingData* training_data, FeedForwardNeuralNetwork::TrainingAlgorithmType algorithm, size_t epochs) {
    std::cout << "Testing feed-forward neural network training with " << TrainingAlgorithmNames[static_cast<int>(algorithm)] << "..." << std::endl;
    nn->SetTrainingAlgorithmType(algorithm);

    std::cout << "\tInitializing weight values to random (-1, 1)..." << std::endl;
    nn->InitializeWeightsRandom();
    std::cout << "\t\tError before training: " << nn->GetError(*training_data) << std::endl;
    nn->Train(training_data, epochs);
    std::cout << "\t\tError after training for " << epochs << " epochs: " << nn->GetError(*training_data) << std::endl;

    std::cout << "\tInitializing weight values via Widrow-Nguyen..." << std::endl;
    nn->InitializeWeights(*training_data);
    std::cout << "\t\tError before training: " << nn->GetError(*training_data) << std::endl;
    nn->Train(training_data, epochs);
    std::cout << "\t\tError after training for " << epochs << " epochs: " << nn->GetError(*training_data) << std::endl << std::endl;
}

void TrainAndTest(FeedForwardNeuralNetwork* nn, TrainingData* training_data, size_t epochs) {
    TrainAndTestNetwork(nn, training_data, FeedForwardNeuralNetwork::TrainingAlgorithmType::Backpropagation, epochs);
    TrainAndTestNetwork(nn, training_data, FeedForwardNeuralNetwork::TrainingAlgorithmType::BatchingBackpropagation, epochs);
    TrainAndTestNetwork(nn, training_data, FeedForwardNeuralNetwork::TrainingAlgorithmType::QuickBackpropagation, epochs);
    TrainAndTestNetwork(nn, training_data, FeedForwardNeuralNetwork::TrainingAlgorithmType::ResilientBackpropagation, epochs);
    TrainAndTestNetwork(nn, training_data, FeedForwardNeuralNetwork::TrainingAlgorithmType::SimulatedAnnealingResilientBackpropagation, epochs);
}

void TestNetwork(RecurrentNeuralNetwork* rnn, TrainingData* training_data, size_t run_steps) {
    std::cout << "Testing recurrent neural network..." << std::endl;

    std::cout << "\tInitializing weight values to random (-1, 1)..." << std::endl;
    rnn->InitializeWeightsRandom();

    std::cout << "\t\tError before running: " << rnn->GetError(*training_data) << std::endl;

    for (size_t i = 0; i < run_steps; i++) {
        // Just run the network forward on each example input.
        for (const auto& example : *training_data) {
            rnn->RunForward(example.input);
        }

        std::cout << "\t\tError after run #" << i+1 << ": " << rnn->GetError(*training_data) << std::endl;
    }

    std::cout << std::endl;
}

int DoTests() {
    TrainingData training_data;

    MakeXorTwoBitTrainingData(&training_data);
    FeedForwardNeuralNetwork nn;
    MakeTestNetwork(&nn, &training_data);
    constexpr size_t test_epochs = 1000;
    TrainAndTest(&nn, &training_data, test_epochs);

    constexpr size_t sine_data_steps = 200;
    MakeSineTrainingData(&training_data, sine_data_steps);
    RecurrentNeuralNetwork rnn;
    MakeTestNetwork(&rnn, &training_data);
    constexpr size_t rnn_run_steps = 10;
    TestNetwork(&rnn, &training_data, rnn_run_steps);

    for (const auto& test : feedForwardTestCases) {
        TestConstruct(test);
    }

    return 0;
}

}  // namespace testing

int main() {
  try {
    testing::DoTests();
  } catch (...) {
    std::cout << "Caught exception running tests.";
    return -1;
  }

  return 0;
}
