//-------------------------------------------------------------------------------------------------------
// Copyright (C) Taylor Woll and panga contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE.txt file in the project root for full license information.
//-------------------------------------------------------------------------------------------------------

#pragma once

#include <vector>

class RandomWrapper;

struct Example {
    std::vector<double> _input;
    std::vector<double> _output;
};

class TrainingData : public std::vector<Example> {
public:
    TrainingData& operator=(const TrainingData& rhs);

    /**
     * Randomize the order of examples in the training data.
     */
    void Shuffle(RandomWrapper* randomWrapper);

    /**
     * Scale with the given algorithm
     */
    void Scale();

    /** 
     * Descale the TrainingData with the scaling algorithm set.
     */
    void Descale();
};
