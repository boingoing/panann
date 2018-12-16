[![master build status](https://travis-ci.org/boingoing/panann.svg?branch=master)](https://travis-ci.org/boingoing/panann/builds#)

# panann

A library for building, running, and training feed-forward artificial neural networks.

## Why panann

Panann is adapted from the neural network component of a hobby project named PAN built way back in 2008. For the tenth anniversary of the project, panann was split out into its own library and re-implemented in c++17. Panann is the PAN artificial neural network (PanAnn) component.

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
