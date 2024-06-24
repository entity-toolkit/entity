conda create -n entity-nompi &&\
conda activate entity-nompi &&\
conda install "conda-forge::gxx[version='>=9,<10']" &&\
conda install conda-forge::hdf5 &&\
conda install conda-forge::adios2 &&\
conda install pip &&\
pip install nt2py

