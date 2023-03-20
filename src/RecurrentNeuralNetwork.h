//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#ifndef RECURRENTNEURALNETWORK_H__
#define RECURRENTNEURALNETWORK_H__

#include "NeuralNetwork.h"

namespace panann {

/**
 * 
 */
class RecurrentNeuralNetwork : public NeuralNetwork {
protected:
    struct LongShortTermMemoryCell {
        /**
         * Index into the |neurons_| vector of the first neuron belonging to the cell.
         */
        size_t neuron_start_index;

        /**
         * Count of neurons belonging to the cell.
         */
        size_t neuron_count;

        /**
         * Index into the |cell_states_| vector of the first state belonging to the cell.
         */
        size_t cell_state_start_index;

        /**
         * Count of cell states belonging to the cell.<br/>
         * Note: All cells currently have the same count of cell states and that count equals |cell_memory_size_|.
         */
        size_t cell_state_count;
    };

public:
    RecurrentNeuralNetwork() = default;
    RecurrentNeuralNetwork(const RecurrentNeuralNetwork&) = delete;
    RecurrentNeuralNetwork& operator=(const RecurrentNeuralNetwork&) = delete;
    ~RecurrentNeuralNetwork() override = default;

    /**
     * Set the number of memory states which each cell will contain.<br/>
     * This value has a quadratic effect on the size of the network topology. Increasing the number of cell memory states increases the number of neurons in each cell gate - effectively, increasing this value by one increases the number of neurons in the network topology by 5 per LSTM cell.
     * Default: 200
     */
    void SetCellMemorySize(size_t memory_size);
    size_t GetCellMemorySize() const;

    void SetHiddenLayerCount(size_t layer_count);

    void RunForward(const std::vector<double>& input) override;

    /**
     * Get a writable vector of memory state for all cells in the network.
     */
    std::vector<double>& GetCellStates();

protected:
    void Allocate() override;
    void ConnectFully() override;

    void UpdateCellState(LongShortTermMemoryCell& cell);

private:
    static constexpr size_t DefaultCellMemorySize = 200;

    std::vector<LongShortTermMemoryCell> cells_;
    std::vector<double> cell_states_;
    size_t cell_memory_size_ = DefaultCellMemorySize;
};

} // namespace panann

#endif  // RECURRENTNEURALNETWORK_H__
