CPU_VENV = qlbm-cpu-venv
GPU_VENV = qlbm-gpu-venv
PYTHON  ?= python3

.PHONY: clean check-python-version install-cpu install-gpu ruff mypy test check-ci

clean:
	rm -rf $(CPU_VENV)
	rm -rf $(GPU_VENV)
	find . -type f -name "*.pyc" -exec rm -f {} \;

check-python-version:
	@PYTHON_VERSION=$$($(PYTHON) --version 2>&1 | awk '{print $$2}'); \
	MAJOR_VERSION=$$(echo $$PYTHON_VERSION | cut -d. -f1); \
	MINOR_VERSION=$$(echo $$PYTHON_VERSION | cut -d. -f2); \
	if [ "$$MAJOR_VERSION" -ne 3 ] || [ "$$MINOR_VERSION" -lt 8 ] || [ "$$MINOR_VERSION" -gt 12 ]; then \
	    echo "Python version must be between 3.8 and 3.12"; \
	    exit 1; \
	fi

install-cpu: check-python-version pyproject.toml
	@ echo "Creating CPU venv..."
	$(PYTHON) -m venv $(CPU_VENV)
	@ echo "Installing dependencies [CPU]..."
	$(CPU_VENV)/bin/python -m pip install --upgrade pip
	$(CPU_VENV)/bin/pip install -e .[cpu,dev]
	@ echo "Installation successful!"

install-gpu: check-python-version pyproject.toml
	@ echo "Creating GPU venv..."
	$(PYTHON) -m venv $(GPU_VENV)
	@ echo "Installing dependencies [GPU]..."
	$(GPU_VENV)/bin/python -m pip install --upgrade pip
	$(GPU_VENV)/bin/pip install -e .[gpu,dev]
	@ echo "Installation successful!"

ruff:
	@ echo Running Ruff...
	$(CPU_VENV)/bin/ruff check qlbm test
	@ echo Ruff was successful.

mypy:
	@ echo Running Mypy...
	$(CPU_VENV)/bin/mypy qlbm test --config-file pyproject.toml
	@ echo Mypy was successful

test:
	@ echo Running pytest...
	$(CPU_VENV)/bin/pytest test/unit --junitxml=pytest_report.xml
	@ echo All tests were successful.

check-ci: ruff mypy test
