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
    nn->AddHiddenLayer(5);
    nn->Construct();
}

void TrainAndTestNetwork(NeuralNetwork* nn, NeuralNetwork::TrainingData* trainingData, NeuralNetwork::TrainingAlgorithmType algorithm, size_t epochs) {
    nn->InitializeWeightsRandom();
    nn->SetTrainingAlgorithmType(algorithm);
    std::cout << "Before training error: " << nn->GetError(trainingData) << std::endl;
    nn->Train(trainingData, epochs);
    std::cout << "After training error: " << nn->GetError(trainingData) << std::endl;
}

int main(int argc, const char** argv) {
    NeuralNetwork::TrainingData trainingData;
    MakeXorTwoBitTrainingData(&trainingData);

    NeuralNetwork nn;
    MakeTestNetwork(&nn, &trainingData);

    TrainAndTestNetwork(&nn, &trainingData, NeuralNetwork::TrainingAlgorithmType::Backpropagation, 1000);
    TrainAndTestNetwork(&nn, &trainingData, NeuralNetwork::TrainingAlgorithmType::BatchingBackpropagation, 1000);
    TrainAndTestNetwork(&nn, &trainingData, NeuralNetwork::TrainingAlgorithmType::QuickBackpropagation, 1000);
    TrainAndTestNetwork(&nn, &trainingData, NeuralNetwork::TrainingAlgorithmType::ResilientBackpropagation, 1000);
    TrainAndTestNetwork(&nn, &trainingData, NeuralNetwork::TrainingAlgorithmType::SimulatedAnnealingResilientBackpropagation, 1000);

    return 0;
}
