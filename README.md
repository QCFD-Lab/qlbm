# `qlbm`

![GitHub License](https://img.shields.io/github/license/qcfd-lab/qlbm?color=%2300A6D6) ![GitHub top language](https://img.shields.io/github/languages/top/qcfd-lab/qlbm?color=%2300A6D6) ![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FQCFD-Lab%2Fqlbm%2Frefs%2Fheads%2Fdev%2Fpyproject.toml?color=%2300A6D6) ![PyPI - Version](https://img.shields.io/pypi/v/qlbm?color=%2300A6D6) ![GitHub commits since latest release](https://img.shields.io/github/commits-since/qcfd-lab/qlbm/latest?color=%2300A6D6) ![GitHub branch check runs](https://img.shields.io/github/check-runs/qcfd-lab/qlbm/main?color=%2300A6D6) <a href="https://arxiv.org/abs/2411.19439">![Static Badge](https://img.shields.io/badge/preprint-blue?style=flat&label=arXiv&color=%2300A6D6)</a>

`qlbm` is a package for the development, simulation, and analysis of **Q**uantum **L**attice **B**oltzmann **M**ethods.

---

`qlbm` is a rapidly evolving, research-oriented piece of software. It contains building blocks for constructing quantum circuits for quantum LBMs and connects these with quantum software infrastructure. `qlbm` is built with end-to-end development environment in mind, including:

- Parsing human-readable `JSON` specifications for QLBMs
- Constructing quantum circuits in [Qiskit](https://www.ibm.com/quantum/qiskit) that implement QLBMs
- Compiling quantum circuits to quantum computer and simulator platforms with Qiskit and [Pytket](https://tket.quantinuum.com/api-docs/)
- Simulating quantum circuits on classical hardware with Qiskit and [Qulacs](http://docs.qulacs.org/en/latest/)
- Visualizing results in [ParaView](https://www.paraview.org/)
- Analyzing the properties , scalability, and performance of quantum algorithms

<p align="center">
<a href="https://qcfd-lab.github.io/qlbm/">
<img width=400 centered alt="Static Badge" src="https://img.shields.io/badge/Documentation-00A6D6%20?style=flat&logo=BookStack&logoColor=%23FFFFFF&logoSize=10&label=Web&color=%2300A6D6&">
</a>
</p>

## PyPI installation

`qlbm` can be installed through `pip`. We recommend the use of a Python 3.12 or 3.13 virtual environment:

```bash
python -m venv qlbm-cpu-venv
pip install --upgrade pip
pip install qlbm
```

Note that `qlbm` evolves quickly and it is likely that the GitHub repository contains new features that the PyPI installation does not. To get the latest developments, we recommend the source installation.

## Install from source

Alternatively, you can install the latest version of `qlbm` by cloning the repository and installing from source as follows (again using Python 3.12 or 3.13):

```bash
git clone https://github.com/QCFD-Lab/qlbm.git
cd qlbm
python -m venv qlbm-cpu-venv
source qlbm-cpu-venv/bin/activate
pip install --upgrade pip
pip install -e .[cpu,dev,docs]
```

If you are using `zsh` (which is the default shell on macOS) you need to replace the last line by

```bash
pip install -e .\[cpu,dev,docs\]
```

We also provide a `make` script for this purpose, which will create the environment from scratch:

```bash
make install-cpu
source qlbm-cpu-venv/bin/activate
```

To override the default Python executable, pass `PYTHON` on the command line:

```bash
make install-cpu PYTHON=your-python-binary
```

## Container installation

The `Docker directory` contains Dockerfiles for running `qlbm` in a containerized environment.

Build the CPU image from the repository root:

```bash
docker build -f ./Docker/build_cpu.Dockerfile -t qlbm-cpu .
```

After the build completes, start an interactive container and mount the local `qlbm` source directory as read-only:

```bash
docker run --rm -it \
  -v "$(pwd)/qlbm:/qlbm/qlbm:ro" \
  qlbm-cpu
```

On systems using SELinux, add the `Z` option to relabel the mounted directory:

```bash
docker run --rm -it \
  -v "$(pwd)/qlbm:/qlbm/qlbm:ro,Z" \
  qlbm-cpu
```

Due to how quickly the code base is evolving, we recommend using the CPU option for stability purposes.

## Algorithms and Usage

Currently, `qlbm` supports the following algorithms:
 - The Quantum Transport Method (Collisionless QLBM) described in [Efficient and fail-safe quantum algorithm for the transport equation](https://doi.org/10.1016/j.jcp.2024.112816) ([arXiv:2211.14269](https://arxiv.org/abs/2211.14269)) by M.A. Schalkers and M. Möller.
 - The Space-Time QLBM/QLGA described in [On the importance of data encoding in quantum Boltzmann methods](https://link.springer.com/article/10.1007/s11128-023-04216-6) by M.A. Schalkers and M. Möller and expanded in [Fully Quantum Lattice Gas Automata Building Blocks for Computational Basis State Encodings](https://doi.org/10.1016/j.jcp.2025.114595).
 - The Linear-encoding Quantum Lattice Gas Automata (LQLGA) described in [On quantum extensions of hydrodynamic lattice gas automata](https://www.mdpi.com/2410-3896/4/2/48) by P. Love and [Fully Quantum Lattice Gas Automata Building Blocks for Computational Basis State Encodings](https://doi.org/10.1016/j.jcp.2025.114595).
 - The Amplitude-Based QLBM with an angle-encoded BGK collision (`ABBGKQLBM`), which adds a $\tau=1$ collision to the amplitude-based encoding by writing the $D_2Q_9$ equilibrium as a linear combination of six trigonometric features of the macroscopic velocity.

The `demos` directory contains several use cases for simulating and analyzing these algorithms. Each demo requires minimal setup once the virtual environment has been configured. Consult the `README.md` file in the `demos` directory for further details.

> **Note on visualization**: we rely on  ParaView for visualizing the flow field of the simulation. You can install ParaView from [this link](https://www.paraview.org/download/).

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
      "shape": "cuboid",
      "x": [18, 20],
      "y": [6, 25],
      "boundary": "specular"
    },
    {
      "shape": "cuboid",
      "x": [23, 25],
      "y": [3, 17],
      "boundary": "bounceback"
    },
    {
      "shape": "cuboid",
      "x": [28, 29],
      "y": [16, 29],
      "boundary": "specular"
    }
  ]
}
```

## Citation

An open access peer-reviewed article describing `qlbm` is available [here](https://doi.org/10.1016/j.cpc.2025.109699). If you use `qlbm`, you can cite it as:

```
@article{georgescu2025qlbm,
title = {qlbm – A quantum lattice Boltzmann software framework},
journal = {Computer Physics Communications},
volume = {315},
pages = {109699},
year = {2025},
issn = {0010-4655},
doi = {https://doi.org/10.1016/j.cpc.2025.109699},
url = {https://www.sciencedirect.com/science/article/pii/S0010465525002012},
author = {C\u{{a}}lin A. Georgescu and Merel A. Schalkers and Matthias M\"{o}ller},
}
```

## Contact

In addition to opening issues, you can contact the developers of `qlbm` at `qcfd-EWI@tudelft.nl`.
