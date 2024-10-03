# qlbm

`qlbm` is a package for the development, simulation, and analysis of **Q**uantum **L**attice **B**oltzmann **M**ethods.



This library contains building blocks for constructing quantum circuits for LBMs and connects the circuits with quantum software infrastructure. `qlbm` provides a platform for an end-to-end development environment, including:

- Parsing human-readable `JSON` specifications for QLBMs
- Constructing quantum circuits in [Qiskit](https://www.ibm.com/quantum/qiskit) that implement QLBMs
- Compiling quantum circuits to quantum computer and simulator platforms with Qiskit and [Pytket](https://tket.quantinuum.com/api-docs/)
- Simulating quantum circuits on classical hardware with Qiskit and [Qulacs](http://docs.qulacs.org/en/latest/)
- Visualizing results in [Paraview](https://www.paraview.org/)
- Analyzing the properties , scalability, and performance of quantum algorithms

`qlbm` is a rapidly evolving, research-oriented piece of software, written almost entirely in Python. A detailed documentation of `qlbm` is available at [here](https://qcfd-lab.github.io/qlbm/).

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

`qlbm` uses quantum circuits to simulate systems that users can specify in simple `JSON` configuration files. For instance, the following configuration describes a 2D system of 64x32 gridpoints, 4 discrete velocities per dimension, and with 3 solid objects placed in the fluid domain:

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