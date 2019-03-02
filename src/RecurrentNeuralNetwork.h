//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#pragma once

#include "NeuralNetwork.h"

namespace panann {

class RecurrentNeuralNetwork : public NeuralNetwork {
protected:
    struct LongShortTermMemoryCell {
        size_t _neuronStartIndex;
        size_t _neuronCount;

        std::vector<double> _cellState;
    };

    std::vector<LongShortTermMemoryCell> _cells;
    size_t _cellMemorySize;

public:
    RecurrentNeuralNetwork();

    void SetCellMemorySize(size_t memorySize);
    size_t GetCellMemorySize();

    void Construct();

    void RunForward(const std::vector<double>* input);

protected:
    void Allocate();
    void ConnectFully();

    void UpdateCellState(size_t cellIndex);
};

} // namespace panann
