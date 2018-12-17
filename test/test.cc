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

using namespace panann;

const char* TrainingAlgorithmNames[] = {
    "Backpropagation",
    "BatchingBackpropagation",
    "QuickBackpropagation",
    "ResilientBackpropagation",
    "SimulatedAnnealingResilientBackpropagation"
};

void MakeXorTwoBitTrainingData(NeuralNetwork::TrainingData* trainingData) {
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

void MakeTestNetwork(NeuralNetwork* nn, NeuralNetwork::TrainingData* trainingData) {
    nn->DisableShortcutConnections();
    nn->SetInputNeuronCount(trainingData->at(0)._input.size());
    nn->SetOutputNeuronCount(trainingData->at(0)._output.size());
    nn->AddHiddenLayer(5);
    nn->AddHiddenLayer(5);
    nn->Construct();
}

void TrainAndTestNetwork(NeuralNetwork* nn, NeuralNetwork::TrainingData* trainingData, NeuralNetwork::TrainingAlgorithmType algorithm, size_t epochs) {
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

void TrainAndTest(NeuralNetwork* nn, NeuralNetwork::TrainingData* trainingData, size_t epochs) {
    TrainAndTestNetwork(nn, trainingData, NeuralNetwork::TrainingAlgorithmType::Backpropagation, epochs);
    TrainAndTestNetwork(nn, trainingData, NeuralNetwork::TrainingAlgorithmType::BatchingBackpropagation, epochs);
    TrainAndTestNetwork(nn, trainingData, NeuralNetwork::TrainingAlgorithmType::QuickBackpropagation, epochs);
    TrainAndTestNetwork(nn, trainingData, NeuralNetwork::TrainingAlgorithmType::ResilientBackpropagation, epochs);
    TrainAndTestNetwork(nn, trainingData, NeuralNetwork::TrainingAlgorithmType::SimulatedAnnealingResilientBackpropagation, epochs);
}

int main(int argc, const char** argv) {
    NeuralNetwork::TrainingData trainingData;
    MakeXorTwoBitTrainingData(&trainingData);

    NeuralNetwork nn;
    MakeTestNetwork(&nn, &trainingData);

    TrainAndTest(&nn, &trainingData, 100000);

    return 0;
}
