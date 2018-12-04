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

int main(int argc, const char** argv) {
    NeuralNetwork nn(100,100);
    nn.AddHiddenLayer(100);
    nn.AddHiddenLayer(200);
    nn.Construct();

    return 0;
}
