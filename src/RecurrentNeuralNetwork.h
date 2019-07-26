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
        size_t _cellStateStartIndex;
    };

    std::vector<LongShortTermMemoryCell> _cells;
    std::vector<double> _cellStates;
    size_t _cellMemorySize;

public:
    RecurrentNeuralNetwork();

    void SetCellMemorySize(size_t memorySize);
    size_t GetCellMemorySize();

    void Construct();

    void RunForward(const std::vector<double>* input);

    /**
     * Get a writable vector of memory state for all cells in the network.
     */
    std::vector<double>* GetCellStates();

protected:
    RecurrentNeuralNetwork(const RecurrentNeuralNetwork&);

    void Allocate();
    void ConnectFully();

    void UpdateCellState(size_t cellIndex);
};

} // namespace panann
