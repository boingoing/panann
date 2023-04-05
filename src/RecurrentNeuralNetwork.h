//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panann contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#ifndef RECURRENTNEURALNETWORK_H__
#define RECURRENTNEURALNETWORK_H__

#include "Perceptron.h"

namespace panann {

/**
 * A recurrent artificial neural network made out of long short term memory cells.<br/>
 *
 * This network doesn't contain ordinary hidden neurons organized into layers. Instead, each layer contains a set of recurrent cells which are each made of a number of hidden units grouped into gates.<br/>
 *
 * Note: RecurrentNeuralNetwork does not support any training algorithms, currently.
 */
class RecurrentNeuralNetwork : public Perceptron {
protected:
    struct LongShortTermMemoryCell {
        /**
         * Index into the |neurons_| vector of the first neuron belonging to the cell.
         */
        size_t neuron_start_index;

        /**
         * Count of neurons belonging to the cell.<br/>
         * Includes neurons in all layers of the cell (but does not include the bias neuron for this cell).
         */
        size_t neuron_count;

        /**
         * Index into the |cell_states_| vector of the first state belonging to the cell.
         */
        size_t cell_state_start_index;

        /**
         * Count of cell states belonging to the cell.<br/>
         * This is also known as the cell memory size.
         */
        size_t cell_state_count;

        size_t GetNeuronsPerGate() const;
        size_t GetForgetGateStartNeuronIndex() const;
        size_t GetInputGateStartNeuronIndex() const;
        size_t GetOutputGateStartNeuronIndex() const;
        size_t GetCandidateCellStateStartNeuronIndex() const;
        size_t GetOutputUnitStartNeuronIndex() const;
    };

    struct CellLayer {
        /**
         * Index into the |cells_| vector of the first cell belonging to this layer.
         */
        size_t cell_start_index;

        /**
         * Count of cells belonging to this layer.
         */
        size_t cell_count;
    };

public:
    RecurrentNeuralNetwork() = default;
    RecurrentNeuralNetwork(const RecurrentNeuralNetwork&) = delete;
    RecurrentNeuralNetwork& operator=(const RecurrentNeuralNetwork&) = delete;
    ~RecurrentNeuralNetwork() override = default;

    /**
     * Set the number of memory states which each cell will contain by default.<br/>
     * This value has a quadratic effect on the size of the network topology. Increasing the number of cell memory states increases the number of neurons in each cell gate - effectively, increasing this value by one increases the number of hidden neurons in the network topology by 5 per LSTM cell.
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
    void AddHiddenLayer(size_t cell_count, const std::vector<size_t>& cell_memory_sizes = {});

protected:
    void ConnectFully() override;
    void FixNeuronConnectionIndices() override;

    void AllocateCellStates();
    bool AreCellStatesAllocated() const;

    void UpdateCellState(const LongShortTermMemoryCell& cell);

    size_t AddCellMemoryStates(size_t count);

    size_t GetCellCount() const;
    LongShortTermMemoryCell& GetCell(size_t index);
    size_t GetCellLayerCount() const;
    CellLayer& GetCellLayer(size_t index);

    /**
     * Initialize all the neurons making up |cell|.<br/>
     * Each gate of the cell will be assigned |input_connection_count| input connections (and zero output connections).<br/>
     * The output layer of the cell will be assigned |output_connection_count| output connections (and zero input connections).
     */
    void InitializeCellNeurons(const LongShortTermMemoryCell& cell, size_t input_connection_count, size_t output_connection_count);

    /**
     * Initialize the neurons in one gate of a cell.
     */
    void InitializeCellNeuronsOneGate(size_t neuron_start_index, size_t neurons_per_gate, ActivationFunctionType activation_function, size_t input_connection_count, size_t output_connection_count);

private:
    static constexpr size_t DefaultCellMemorySize = 200;

    std::vector<CellLayer> layers_;
    std::vector<LongShortTermMemoryCell> cells_;
    std::vector<double> cell_states_;
    size_t cell_states_count_ = 0;
    size_t cell_memory_size_ = DefaultCellMemorySize;
    bool is_allocated_ = false;
};

} // namespace panann

#endif  // RECURRENTNEURALNETWORK_H__
