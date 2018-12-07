//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panga contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#include <vector>
#include <iostream>
#include <cassert>
#include <cstring>
#include <sstream>

#include "NeuralNetwork.h"
#include "TrainingData.h"

int main(int argc, const char** argv) {
    TrainingData trainingData;
    trainingData.resize(4);
    trainingData[0]._input = { 0.0, 0.0 };
    trainingData[0]._output = { 0.0 };
    trainingData[1]._input = { 1.0, 1.0 };
    trainingData[1]._output = { 0.0 };
    trainingData[2]._input = { 1.0, 0.0 };
    trainingData[2]._output = { 1.0 };
    trainingData[3]._input = { 0.0, 1.0 };
    trainingData[3]._output = { 1.0 };

    NeuralNetwork nn;
    nn.SetTrainingAlgorithmType(NeuralNetwork::TrainingAlgorithmType::SimulatedAnnealingResilientBackpropagation);
    nn.SetInputNeuronCount(2);
    nn.SetOutputNeuronCount(1);
    nn.AddHiddenLayer(5);
    nn.AddHiddenLayer(5);
    nn.AddHiddenLayer(5);
    nn.Construct();
    nn.InitializeWeightsRandom();

    std::cout << "Before training error: " << nn.GetError(&trainingData) << std::endl;
    nn.Train(&trainingData, 10000);

    std::cout << "After training error: " << nn.GetError(&trainingData) << std::endl;

    return 0;
}
