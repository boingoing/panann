//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#ifndef RECURRENTNEURALNETWORK_H__
#define RECURRENTNEURALNETWORK_H__

#include "NeuralNetwork.h"

namespace panann {

/**
 * A recurrent artificial neural network made out of long short term memory cells.<br/>
 * 
 * This network doesn't contain ordinary hidden neurons organized into layers. Instead, each layer contains a set of recurrent cells which are each made of a number of hidden units grouped into gates.
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
         * Count of cell states belonging to the cell.
         */
        size_t cell_state_count;
    };

    struct CellLayer {
        size_t cell_start_index;
        size_t cell_count;
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

    void RunForward(const std::vector<double>& input) override;

    /**
     * Get a writable vector of memory state for all cells in the network.
     */
    std::vector<double>& GetCellStates();

    void AddHiddenLayer(size_t neuron_count) = delete;

    /**
     * Add a hidden layer of LSTM cells.<br/>
     * Each cell may have a different cell memory size passed via |cell_memory_sizes|. If the vector doesn't contain an element for a cell or if that element is 0, the cell memory size for that cell will be the default returned via GetCellMemorySize().
     * @see GetCellMemorySize()
    */
    void AddHiddenLayer(size_t cell_count, const std::vector<size_t>& cell_memory_sizes);

protected:
    void Allocate() override;
    void ConnectFully() override;

    void UpdateCellState(const LongShortTermMemoryCell& cell);

private:
    static constexpr size_t DefaultCellMemorySize = 200;

    std::vector<CellLayer> layers_;
    std::vector<LongShortTermMemoryCell> cells_;
    std::vector<double> cell_states_;
    size_t cell_states_count_ = 0;
    size_t cell_memory_size_ = DefaultCellMemorySize;
};

} // namespace panann

#endif  // RECURRENTNEURALNETWORK_H__
