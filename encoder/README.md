Fast encoder for RS that utilizes LUT for small finite fields and carryless
multiply instructions and properties of modular arithmetic to significantly
accelerate 32 bit fields.

# Installation

Ensure the latest version of the rust toolchain and a compiler are installed.
Recommended installation method for rust is [rustup](https://rustup.rs/), for
the c compiler you should use your system package manager. Python and pip are also required. 

One all dependencies are installed you can install the package as follows:

``` shell
cd encoder
pip install .
```

# Usage

``` python

from encoder import Encoder, Algorithm, SymbolSize

encoder = Encoder(128, SymbolSize.G2x8, Algorithm.ReedSalomon)
encoder.generate()
collions = encoder.encode(255)
print(f"collions {collions}")

``` 