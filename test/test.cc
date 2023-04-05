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
#include "RecurrentNeuralNetwork.h"
#include "TrainingData.h"

namespace panann {

const char* TrainingAlgorithmNames[] = {
    "Backpropagation",
    "BatchingBackpropagation",
    "QuickBackpropagation",
    "ResilientBackpropagation",
    "SimulatedAnnealingResilientBackpropagation"
};

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

void TrainAndTestNetwork(FeedForwardNeuralNetwork* nn, TrainingData* training_data, FeedForwardNeuralNetwork::TrainingAlgorithmType algorithm, size_t epochs) {
    std::cout << "Testing training with " << TrainingAlgorithmNames[static_cast<int>(algorithm)] << "..." << std::endl;
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

    return 0;
}

}  // namespace panann

int main() {
    return panann::DoTests();
}
