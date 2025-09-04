# Container files

We provide two docker files fo `qlbm`: one for regular development and one for GPU integration.

## Development container

`build_cpu.Dockerfile` can be used to build a container image that has all dependencies of `qlbm` for active development. You can build the image from the repository root as follows:

```bash
docker build -t qlbm -f Docker/build_cpu.Dockerfile .
```

Once the image is built, it will have all the dependencies, but not `qlbm` itself. You can actively link the repository files to the running container with the command

```bash
docker run -it --volume $(pwd):/qlbm/ qlbm
```

Once inside the container, you can quickly link the files in the volume to `pip` with

```
pip install -e .[cpu,dev,docs]
```

This will allow you to edit the files on your machine's file system in a text editor of your choice, and have the updates immediately available in the running container, without reinstalling anything.

## GPU container

`build_gpu.Dockerfile` contains an image allows you to simulate `qlbm` algorithms on nVidia GPU hardware with Qiskit and using the nVidia cuQuantum appliance.
