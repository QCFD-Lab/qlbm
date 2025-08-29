# Contributing

Pull requests are welcome. To make a PR, first fork the repo, make your proposed changes on the main branch, and open a PR from your fork. If it passes tests and is accepted after review, it will be merged. For discussions about feature requests and contributions, please contact `qcfd-EWI@tudelft.nl`.

## Code style

### Formatting and Linting

All code should be formatted using `ruff` with default options. You should verify that your changes adhere to the `ruff` standards before submitting a PR. The exact version used in the CI pipeline can be found in the `pyproject.toml` file. Both the source code `ruff check qlbm` and the demos `ruff check demos` should adhere to these standards.

### Type annotation

`mypy` is used as a static type checker and all submissions must pass its checks. YYou should verify that your changes adhere to the `mypy` standards before submitting a PR.  The exact version used in the CI pipeline can be found in the `pyproject.toml` file. There are some custom rules used for type checking: you can check whether your submission adheres to them by running `mypy qlbm test --config-file pyproject.toml`.
Linting


### Tests

We encourage each contribution to include unit tests for the added functionality. Tests reside in the `test` directory and can be executed using `pytest test/unit`.
Please sure all tests are passing before submitting a PR, and verify that simple flow simulations work as intended.
When fixing a bug, please add a test that demonstrates the fix.

### Documentation

We encourage all contributions to contain an adequate and thorough documentation of the changes. All contributed classes, methods, and attributes should be documented using the established style. Where appropriate, please include code-block examples and references to the scientific literature.

We use `sphinx` to automatically generate our documentation website. To make sure your contribution fits the standards of the code base, you can execute `make docs` in the root directory. To check the documentation website locally, you can first `cd docs` and then `make clean html` to build the website locally. You can view the updated web pages by opening `docs/build/html/index.html`.

### Easy verification

To verify that your changes comply with our checks, you can simply run `make check-ci` to run all checks.