# qlbm

`qlbm` is a package for the development, simulation, and analysis of **Q**uantum **L**attice **B**oltzmann **M**ethods.



This library contains building blocks for constructing quantum circuits for LBMs and connects the circuits with quantum software infrastructure. `qlbm` provides a platform for an end-to-end development environment, including:

- Parsing human-readable `JSON` specifications for QLBMs
- Constructing quantum circuits in [Qiskit](https://www.ibm.com/quantum/qiskit) that implement QLBMs
- Compiling quantum circuits to quantum computer and simulator platforms with Qiskit and [Pytket](https://tket.quantinuum.com/api-docs/)
- Simulating quantum circuits on classical hardware with Qiskit and [Qulacs](http://docs.qulacs.org/en/latest/)
- Visualizing results in [Paraview](https://www.paraview.org/)
- Analyzing the properties , scalability, and performance of quantum algorithms

`qlbm` is a rapidly evolving, research-oriented piece of software, written almost entirely in Python.

## PyPI installation

`qlbm` can be installed through `pip`. We recommend th euse of a **Python 3.11** virtual environment:

```bash
python3.11 -m venv qlbm-cpu-venv
pip install --upgrade pip
pip install qlbm
```

## Local installation

To install from source, first clone the repository:

```bash
git clone git@github.com:QCFD-Lab/qlbm.git
```

We again recommend to install and use the current version of `qlbm` in a **Python 3.11** virtual environment. To set up the virtual environment, you can run following installation:

```bash
python3.11 -m venv qlbm-cpu-venv
source qlbm-cpu-venv/bin/activate
pip install -e .[cpu,dev]
```

Alternatively, you can use the `make` script provided for this purpose, which will create the environment from scratch as well:

```
make install-cpu
source qlbm-cpu-venv/bin/activate
```

`qlbm` additionally supports several other options, including GPU and MPI simulation. There are also Docker container images in the `Docker` directory. Due to how quickly the code base is evolving, we recommend using the CPU option for stability purposes.

## Algorithms and Usage


Currently, `qlbm` supports two algorithms:
 - The Collisionless QLBM described in [Efficient and fail-safe quantum algorithm for the transport equation](https://doi.org/10.1016/j.jcp.2024.112816) ([arXiv:2211.14269](https://arxiv.org/abs/2211.14269)) by M.A. Schalkers and M. Möller.
 - The Space-Time QLBM described in [On the importance of data encoding in quantum Boltzmann methods](https://link.springer.com/article/10.1007/s11128-023-04216-6) by M.A. Schalkers and M. Möller.

The `demos` directory contains several use cases for simulating and analyzing these algorithms. Each demo requires minimal setup once the virtual environment has been configured. Consult the `README.md` file in the `demos` directory for further details.

> **Note on visualization**: we rely on  Paraview for visualizing the flow field of the simulation. You can install Paraview from [this link](https://www.paraview.org/download/).

## Configuration

`qlbm` simulates systems that users can specify in simple `JSON` configuration files. For instance, the following configuration describes a 2D system of 64x32 gridpoints, 4 discrete velocities per dimension, and with 3 solid objects placed in the fluid domain:

```JSON
{
  "lattice": {
    "dim": {
      "x": 64,
      "y": 32
    },
    "velocities": {
      "x": 4,
      "y": 4
    }
  },
  "geometry": [
    {
      "x": [18, 20],
      "y": [6, 25],
      "boundary": "specular"
    },
    {
      "x": [23, 25],
      "y": [3, 17],
      "boundary": "bounceback"
    },
    {
      "x": [28, 29],
      "y": [16, 29],
      "boundary": "specular"
    }
  ]
}
```

System specifications are constrained by several restrictions:

- The number of grid points in each dimension and the number of discrete velocities must be powers of 2.
- Solid obstacles must not overlap, and should be at least two gridpoints apart in each dimension.
- The initial conditions must not place any fluid particle populations inside any object.
- Only `specular` and `bounceback` boundary conditions are supported at the moment.

The `demos/lattices` directory contains examples of additional lattice specifications that can be used for benchmarking algorithms.


## Structure

The structure of the source code is the following:

```
qlbm
├── analysis
├── components/
│   ├── base.py
|   └── common/
│       └── ...
│   ├── collisionless/
│   │   ├── streaming.py
│   │   ├── specular_reflection.py
│   │   └── ...
│   └── spacetime/
│       └── ...
├── infra/
│   ├── result/
│   │   ├── base.py
│   │   ├── collisionless_result.py
│   │   └── spacetime_result.py
│   ├── runner/
│   │   ├── base.py
│   │   └── mpi_qulacs.py
│   ├── reinitializer/
│   │   ├── base.py
│   │   ├── collisionless_reinitializer.py
│   │   └── spacetime_reinitializer.py
│   └── compiler.py
├── lattice/
│   ├── lattices/
│   │   ├── base.py
│   │   ├── collisionless_lattice.py
│   │   └── spacetime_lattice.py
│   └── blocks.py
└── tools
```

The `components` directory contains the quantum circuit generation building blocks organized in increasing order of complexity: _primitives_, _operators_, and end-to-end _algorithms_. The `components/base.py` file contains the base classes for each type of component, while the `collisionless` and `spacetime` and subdirectories contain algorithm-specific counterparts.
Many of the classes of `qlbm` contain algorithmic-specific implementations. The two algorithms currently implemented in `qlbm` are theoretically described in the papers _[Efficient and fail-safe quantum algorithm for the transport equation](https://doi.org/10.1016/j.jcp.2024.112816)_ ([arXiv:2211.14269](https://arxiv.org/abs/2211.14269)) and _[On the importance of data encoding in quantum Boltzmann methods](https://link.springer.com/article/10.1007/s11128-023-04216-6)_.
The `lattice` directory contains the classes that encode the properties of the lattice under simulation, as well as geometry data. This directory is again split between a `base`, a `collisionless_lattice` and a `spacetime_lattice` implementation.
The `infra` directory holds the code that interfaces with surrounding infrastructure, namely compilers and simulators. The `tools` directory contains miscellaneous utilities under the `exceptions.py` and `utils.py` files.
