# .PHONY defines parts of the makefile that are not dependant on any specific file
# This is most often used to store functions
.PHONY = init format format_check format_notebooks lint type_check all

init:
	@echo "Create a development python environment"
	python3 -m pip install virtualenv
	python3 -m venv ptp_dev
	. ptp_dev/bin/activate && cat requirements.txt | xargs -n 1 pip install
	. ptp_dev/bin/activate && pip install -r requirements_dev.txt

format:
	@echo "Format code according to isort"
	. ptp_dev/bin/activate && isort *.py
	@echo "Format code according to black"
	. ptp_dev/bin/activate && black *.py

format_check:
	@echo "Check code format according to isort"
	. ptp_dev/bin/activate && isort *.py --check
	@echo "Check code format according to black"
	. ptp_dev/bin/activate && black *.py --check

format_notebooks:
	@echo "Format notebooks according to isort"
	. ptp_dev/bin/activate && nbqa isort .
	@echo "Format notebooks according to black"
	. ptp_dev/bin/activate && nbqa black .

lint:
	@echo "Linter check: Flake8"
	. ptp_dev/bin/activate && flake8 .

type_check:
	@echo "Type-test check: mypy"
	. ptp_dev/bin/activate && mypy .

all: format format_notebooks lint type_check 
