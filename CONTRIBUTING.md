# How to contribute

## Environment set up

1. Install [anaconda](https://conda.io/miniconda.html)/[miniconda](https://conda.io/miniconda.html)
2. Create the torchvideo environment:

   ```bash
   $ conda env create -n torchvideo -f environment.yml
   $ conda activate torchvideo
   ```
3. Check everything is installed properly by running the tests:
   ```bash
   $ make test
   ```
4. Check you can build the documentation
   ```bash
   $ make docs

   # For linux:
   $ xdg-open docs/build/html/index.html
   # For macOS:
   $ open docs/build/html/index.html
   ```
5. Set up pre-commit hooks:
   ```bash
   $ pip install pre-commit
   $ pre-commit install
   ```
   These will run every time you commit and ensure the code type checks, doesn't
   have trailing whitespace, runs the black formatter etc.

## Making changes

* Ensure docstrings are [Google
    style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
* Ensure changes have automated tests
* Ensure there are documentation updates if necessary
* Add changes to [`CHANGELOG.md`](/CHANGELOG.md)


## Adding features

### Dataset

* Implement your dataset with tests in `src/torchvideo/datasets`.
* Add a new entry `docs/source/datasets.rst` of the form:
  ```rst
  MyDataset
  ~~~~~~~~~
  .. autoclass:: MyDataset
      :special-members: __getitem__,__len__
  ```
* Add an example of usage to `examples/datasets.ipynb`

### Transform

* Implement your transformation with tests in `src/torchvideo/transforms`,
  ideally splitting it into a pure functional core, and a class that calls this.
* Add a new entry `docs/source/transforms.rst` of the form:
  ```rst
  MyTransform
  ~~~~~~~~~~~
  .. autoclass:: MyTransform
      :special-members: __call__
  ```
* Add an example of usage to `examples/transforms.ipynb`

### Sampler

* Implement your sampler with tests in `src/torchvideo/samplers`.
* Add a new entry `docs/source/samplers.rst` of the form:
  ```rst
  MySampler
  ~~~~~~~~~
  .. autoclass:: MySampler
  ```
* Add an example of usage to `examples/samplers.ipynb`
