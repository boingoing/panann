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
        std::vector<double> _hiddenState;
        std::vector<double> _cellState;

        size_t _neuronStartIndex;
        size_t _neuronCount;
    };

    std::vector<LongShortTermMemoryCell> _cells;
    size_t _cellMemorySize;

public:
    void SetCellMemorySize(size_t memorySize);
    size_t GetCellMemorySize();

    void Allocate();
    void ConnectFully();
    void Construct();

    void UpdateCellState(LongShortTermMemoryCell* cell);
};

} // namespace panann
