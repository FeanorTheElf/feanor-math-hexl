# `feanor-math-hexl`

A wrapper around the [Intel Homomorphic Encryption Accelerator (HEXL)](https://github.com/intel/hexl) library that provides a fast implementation of power-of-two length negacyclic Number-Theoretic transforms (NTT).

The library builds on [`feanor-math`](https://crates.io/crates/feanor-math), and exposes the base wrapper [`hexl::HEXLNegacyclicNTT`] and also the convolution algorithm [`conv::HEXLConvolution`] that uses it internally.

## How to build

To use this library, you need an installation of HEXL.
The buildscript will not attempt to download or build HEXL, thus you must ensure that the HEXL header files and a static library build are available.
This library will search for these in a few standard locations (currently only `/usr/local/include` resp. `/usr/local/lib`), but the location can be overriden using environment variables.
 - setting `HEXL_DIR` will result in `feanor-math-hexl` expecting the static library to be in `HEXL_DIR/lib` and the headers in `HEXL_DIR/include`
 - setting `HEXL_LIB_DIR` and `HEXL_INCLUDE_DIR` will result in `feanor-math-hexl` expecting the static library to be in `HEXL_LIB_DIR` and the headers in `HEXL_INCLUDE_DIR`
