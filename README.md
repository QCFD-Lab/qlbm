# `qlbm`

![GitHub License](https://img.shields.io/github/license/qcfd-lab/qlbm?color=%2300A6D6) ![GitHub top language](https://img.shields.io/github/languages/top/qcfd-lab/qlbm?color=%2300A6D6) ![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FQCFD-Lab%2Fqlbm%2Frefs%2Fheads%2Fdev%2Fpyproject.toml?color=%2300A6D6) ![PyPI - Version](https://img.shields.io/pypi/v/qlbm?color=%2300A6D6) ![GitHub commits since latest release](https://img.shields.io/github/commits-since/qcfd-lab/qlbm/latest?color=%2300A6D6) ![GitHub branch check runs](https://img.shields.io/github/check-runs/qcfd-lab/qlbm/main?color=%2300A6D6) <a href="https://arxiv.org/abs/2411.19439">![Static Badge](https://img.shields.io/badge/preprint-blue?style=flat&label=arXiv&color=%2300A6D6)</a>

`qlbm` is a package for the development, simulation, and analysis of **Q**uantum **L**attice **B**oltzmann **M**ethods.

---

`qlbm` is a rapidly evolving, research-oriented piece of software. It contains building blocks for constructing quantum circuits for quantum LBMs and connects these with quantum software infrastructure. `qlbm` is built with end-to-end development environment in mind, including:

- Parsing human-readable `JSON` specifications for QLBMs
- Constructing quantum circuits in [Qiskit](https://www.ibm.com/quantum/qiskit) that implement QLBMs
- Compiling quantum circuits to quantum computer and simulator platforms with Qiskit and [Pytket](https://tket.quantinuum.com/api-docs/)
- Simulating quantum circuits on classical hardware with Qiskit and [Qulacs](http://docs.qulacs.org/en/latest/)
- Visualizing results in [Paraview](https://www.paraview.org/)
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

## Local installation

Alternatively, you can also install the latest version of `qlbm` by cloning the repository and installing from source as follows (again using Python 3.12 or 3.13):

```bash
git clone git@github.com:QCFD-Lab/qlbm.git
cd qlbm
python -m venv qlbm-cpu-venv
source qlbm-cpu-venv/bin/activate
pip install --upgrade pip
pip install -e .[cpu,dev,docs]
```
We also provide a `make` script for this purpose, which will create the environment from scratch:

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

A preprint describing `qlbm` in detail is currently available on [arXiv](https://arxiv.org/abs/2411.19439). If you use `qlbm`, you can cite it as per the [CITATION.bib file](CITATION.bib).