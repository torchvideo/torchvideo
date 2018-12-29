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

.PHONY: function_test
functional_test:
	COVERAGE_FILE=.coverage-functional PYTHONPATH=$(PYTHONPATH) pytest tests/functional

.PHONY: docs
docs:
	$(MAKE) -C docs html
