PYTHONPATH := "src:tests"

.PHONY: test
test: unit_test functional_test
	coverage combine .coverage-*
	coverage xml
	coverage html
	coverage report

.PHONY: unit_test
unit_test:
	COVERAGE_FILE=.coverage-unit PYTHONPATH=$(PYTHONPATH) pytest tests/unit

.PHONY: functional_test
functional_test:
	COVERAGE_FILE=.coverage-functional PYTHONPATH=$(PYTHONPATH) pytest tests/functional

.PHONY: doctest
doctest:
	COVERAGE_FILE=.coverage-doctest PYTHONPATH=$(PYTHONPATH) pytest src


.PHONY: docs
docs:
	$(MAKE) -C docs html

.PHONY: mypy
mypy:
	mypy src/torchvideo

.PHONY: clean
clean: clean-docs clean-build


.PHONY: clean-docs
clean-docs:
	$(MAKE) -C docs clean

.PHONY: clean-build
clean-build:
	@rm -rf dist build
