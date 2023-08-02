# pySZe

A python package to model the Sunyaez-Zeldovich (SZ) effect observations.

### PREREQUISITES

1. [Numpy](http://www.numpy.org/)
2. [Scipy](https://scipy.org/install.html)
3. [Astropy](https://www.astropy.org/)
4. [tqdm](https://github.com/tqdm/tqdm)

## INSTALLATION

To install the package from source, one should clone this package running the following::

    git clone https://github.com/sambit-giri/pySZe.git

To install the package in the standard location, run the following in the root directory::

    python setup.py install

In order to install it in a separate directory::

    python setup.py install --home=directory

One can also install the latest version using pip by running the following command::

    pip install git+https://github.com/sambit-giri/pySZe.git

The dependencies should be installed automatically during the installation process. If they fail for some reason, you can install them manually before installing pySZe. The list of required packages can be found in the requirements.txt file present in the root directory.

### Tests

For testing, one can use [pytest](https://docs.pytest.org/en/stable/). It can be installed using pip. To run all the test script, run the either of the following::

    python -m pytest tests

## CONTRIBUTING

If you find any bugs or unexpected behavior in the code, please feel free to open a [Github issue](https://github.com/sambit-giri/pySZe/issues). The issue page is also good if you seek help or have suggestions for us.