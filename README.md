[![master build status](https://travis-ci.org/boingoing/panann.svg?branch=master)](https://travis-ci.org/boingoing/panann/builds#)

# panann

A library for building, running, and training feed-forward artificial neural networks.

## Why panann

Panann is adapted from the neural network component of a hobby project named PAN built way back in 2008. For the tenth anniversary of the project, panann was split out into its own library and re-implemented in c++17. Panann is the PAN artificial neural network (PanAnn) component.

## Using panann

Panann has a simple interface for constructing and running artificial neural networks.

```c++
using namespace panann;

// Make XOR 2-bit training set.
TrainingData data;
data.resize(4);
data[0]._input = { 0, 0 };
data[0]._output = { 0 };
data[1]._input = { 1, 1 };
data[1]._output = { 0 };
data[2]._input = { 1, 0 };
data[2]._output = { 1 };
data[3]._input = { 0, 1 };
data[3]._output = { 1 };

// Build a simple, multi-layer feed-forward network.
NeuralNetwork nn;
nn.SetInputNeuronCount(2);
nn.SetOutputNeuronCount(1);
nn.AddHiddenLayer(5);
nn.AddHiddenLayer(5);
nn.Construct();
nn.SetTrainingAlgorithmType(NeuralNetwork::TrainingAlgorithmType::SimulatedAnnealingResilientBackpropagation);

nn.InitializeWeightsRandom();
std::cout << "Error before training: " << nn.GetError(&data) << std::endl;

nn.Train(&data, 100000);
std::cout << "Error after training for 100000 epochs: " << nn.GetError(&data) << std::endl;
```

## Building panann

You can build panann on any platform with a compiler which supports c++17 language standards mode. The library is designed to be portable and easy to add to your project. Add the panann source files in `panann/src` to your build definition and you should be ready to use panann.

### Tested build configurations

Windows 10
* CMake 3.13.0-rc3
* Visual Studio 2017 15.8.9

Ubuntu 18.04
* CMake 3.10.2
* Clang 6.0.0

## Testing panann

The library ships with a simple test program in the `panann/test` folder.

```console
> git clone https://github.com/boingoing/panann/panann.git
> cd panann/out
> cmake ..
> make
> ./panann_test
```

### Using Visual Studio on Windows

Above `cmake` command generates a Visual Studio solution file (`panann/out/panann_test.sln`) on Windows platforms with Visual Studio. You can open this solution in Visual Studio and use it to build the test program.

## Documentation

https://boingoing.github.io/panann/html/annotated.html
