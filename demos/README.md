# `qlbm` demos

This directory contains contains several examples of how users can use `qlbm` for simulating, visualizing, and analyzing QLBM algorithms. We recommend that each notebook is executed in a **python3.12** or **python3.13** virtual environment from this (`demos`) directory. You can create a new virtual environment as follows:

```bash
python -m venv qlbm-cpu-venv
source qlbm-cpu-venv/bin/activate
mkdir qlbm-output
pip install --upgrade pip
pip install qlbm jupyter ipykernel matplotlib seaborn pandas
jupyter-lab
```

 Currently, the following directories are available:

## Visualization

Users can visualize both geometry and quantum circuits constructed from the JSON configurations. All quantum components (primitives, operators, algorithms) can be visually serialized to matplotlib, Latex, or text formats with the notebooks in the `visualization` directory.

## Simulation

`qlbm` currently supports two algorithms with different configurations and capabilities. To simulate the end-to-end algorithms and visualize the resulting flow field, users can interact with the notebooks in the `simulation` directory. Each notebook will generate outputs in a new `qlbm-output` directory. Each output subdirectory of `qlbm-output` will itself contain a `paraview` directory where the time step `step_<x>.vti` and geometry `cube_<x>.stl` files can be visualized in Paraview.

> **Note on visualization**: we rely on  Paraview for visualizing the flow field of the simulation. You can install Paraview from [this link](https://www.paraview.org/download/).

## Benchmarks

Users who are interested in developing and analyzing the theoretical and runtime properties of QLBMs can use the notebooks residing in the `benchmarks` directory. These notebooks contain utilities for automatically generating QLBM algorithms of different scales, logging relevant data about their properties, and parsing the data into helpful formats such as in-memory databases or plots. Benchmarks include simulator performance, statevector snapshot performance, and algorithm scalability with real hardware.