# neutral-atom-image-analysis
Neutral Atom Image Analysis

## Acknowledgement
Built upon [David Wei's projection-based state reconstruction](https://github.com/david-wei/state_reconstruction) and [my performance improvements upon that](https://github.com/jonaswinklmann/state_reconstruction_performance).
Still uses this library and its dependencies during calibration.

## Dependencies
* [state_reconstruction](https://github.com/david-wei/state_reconstruction) or [state_reconstruction_(performance)](https://github.com/jonaswinklmann/state_reconstruction_performance)
* [libics](https://www.github.com/david-wei/libics)
* [Eigen](https://gitlab.com/libeigen/eigen) (via submodule, no installation required)
* [pybind11](https://github.com/pybind/pybind11) (via submodule, no installation required)

## Installation
Currently, only Linux is explicitly supported.
After pulling the repository, run ```git submodule update --init --recursive``` to initialize all submodules

Start in the neutral_atom_image_analysis folder. To compile the C++ library
```
cd neutral_atom_image_analysis_cpp/
make <all/fresh>
cd ../
```
To install the library:
```
pip install .
```

The script ```compileCppAndInstallPip.sh``` combines building of the C++ library and pip installation.
