//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#include <vector>
#include <iostream>
#include <cassert>
#include <cstring>
#include <sstream>

#include "NeuralNetwork.h"
#include "TrainingData.h"
#include "RecurrentNeuralNetwork.h"

using namespace panann;

const char* TrainingAlgorithmNames[] = {
    "Backpropagation",
    "BatchingBackpropagation",
    "QuickBackpropagation",
    "ResilientBackpropagation",
    "SimulatedAnnealingResilientBackpropagation"
};

void MakeXorTwoBitTrainingData(TrainingData* trainingData) {
    trainingData->resize(4);
    trainingData->at(0)._input = { 0.0, 0.0 };
    trainingData->at(0)._output = { 0.0 };
    trainingData->at(1)._input = { 1.0, 1.0 };
    trainingData->at(1)._output = { 0.0 };
    trainingData->at(2)._input = { 1.0, 0.0 };
    trainingData->at(2)._output = { 1.0 };
    trainingData->at(3)._input = { 0.0, 1.0 };
    trainingData->at(3)._output = { 1.0 };
}

void MakeSineTrainingData(TrainingData* trainingData, size_t steps) {
    size_t sineSampleCount = steps * 2;
    double stepSize = 1.0 / sineSampleCount;
    constexpr double pi = 3.14159265358979323846;
    std::vector<double> sineData;
    sineData.resize(sineSampleCount);

    for (size_t i = 0; i < sineSampleCount; i++) {
        sineData[i] = std::sin(stepSize * i * 2.0 * pi);
    }

    trainingData->FromSequentialData(&sineData, steps);
}

void MakeTestNetwork(NeuralNetwork* nn, TrainingData* trainingData) {
    nn->DisableShortcutConnections();
    nn->SetInputNeuronCount(trainingData->at(0)._input.size());
    nn->SetOutputNeuronCount(trainingData->at(0)._output.size());
    nn->AddHiddenLayer(5);
    nn->AddHiddenLayer(5);
    nn->Construct();
}

void MakeTestNetwork(RecurrentNeuralNetwork* rnn, TrainingData* trainingData) {
    rnn->SetInputNeuronCount(trainingData->at(0)._input.size());
    rnn->SetOutputNeuronCount(trainingData->at(0)._output.size());
    rnn->AddHiddenLayer(5);
    rnn->SetCellMemorySize(5);
    rnn->Construct();
}

void TrainAndTestNetwork(NeuralNetwork* nn, TrainingData* trainingData, NeuralNetwork::TrainingAlgorithmType algorithm, size_t epochs) {
    std::cout << "Testing training with " << TrainingAlgorithmNames[(int)algorithm] << "..." << std::endl;
    nn->SetTrainingAlgorithmType(algorithm);

    std::cout << "\tInitializing weight values to random (-1, 1)..." << std::endl;
    nn->InitializeWeightsRandom();
    std::cout << "\t\tError before training: " << nn->GetError(trainingData) << std::endl;
    nn->Train(trainingData, epochs);
    std::cout << "\t\tError after training for " << epochs << " epochs: " << nn->GetError(trainingData) << std::endl;

    std::cout << "\tInitializing weight values via Widrow-Nguyen..." << std::endl;
    nn->InitializeWeights(trainingData);
    std::cout << "\t\tError before training: " << nn->GetError(trainingData) << std::endl;
    nn->Train(trainingData, epochs);
    std::cout << "\t\tError after training for " << epochs << " epochs: " << nn->GetError(trainingData) << std::endl;
}

void TrainAndTest(NeuralNetwork* nn, TrainingData* trainingData, size_t epochs) {
    TrainAndTestNetwork(nn, trainingData, NeuralNetwork::TrainingAlgorithmType::Backpropagation, epochs);
    TrainAndTestNetwork(nn, trainingData, NeuralNetwork::TrainingAlgorithmType::BatchingBackpropagation, epochs);
    TrainAndTestNetwork(nn, trainingData, NeuralNetwork::TrainingAlgorithmType::QuickBackpropagation, epochs);
    TrainAndTestNetwork(nn, trainingData, NeuralNetwork::TrainingAlgorithmType::ResilientBackpropagation, epochs);
    TrainAndTestNetwork(nn, trainingData, NeuralNetwork::TrainingAlgorithmType::SimulatedAnnealingResilientBackpropagation, epochs);
}

int main(int argc, const char** argv) {
    TrainingData trainingData;
    MakeSineTrainingData(&trainingData, 200);

    RecurrentNeuralNetwork rnn;
    MakeTestNetwork(&rnn, &trainingData);
    rnn.InitializeWeightsRandom(-1.0, 1.0);

    rnn.RunForward(&trainingData[0]._input);
    double e1 = rnn.GetError(&trainingData[0]._output);
    rnn.RunForward(&trainingData[0]._input);
    double e2 = rnn.GetError(&trainingData[0]._output);
    rnn.RunForward(&trainingData[0]._input);
    double e3 = rnn.GetError(&trainingData[0]._output);
    rnn.RunForward(&trainingData[0]._input);
    double e4 = rnn.GetError(&trainingData[0]._output);
    rnn.RunForward(&trainingData[0]._input);
    double e5 = rnn.GetError(&trainingData[0]._output);

    MakeXorTwoBitTrainingData(&trainingData);

    NeuralNetwork nn;
    MakeTestNetwork(&nn, &trainingData);

    TrainAndTest(&nn, &trainingData, 1000);

    return 0;
}
