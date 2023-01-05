from setuptools import setup


setup(\
        name="yuca", \
        packages=["yuca"], \
        version = "0.0", \
        description = "Your Universal Cellular Automata", \
        install_requires=[\
                        "bokeh==2.4.1",\
                        "coverage==7.0.3",\
                        "jupyter==1.0.0",\
                        "notebook>=6.4.12",\
                        "numpy==1.22.0",\
                        "torch==1.13.1",\
                        "cycler==0.11.0",\
                        "decorator==4.4.2",\
                        "future==0.18.2",\
                        "imageio==2.14.1",\
                        "kiwisolver==1.3.1",\
                        "matplotlib==3.3.4",\
                        "networkx==2.5.1",\
                        "Pillow==9.3.0",\
                        "pyparsing==3.0.7",\
                        "python-dateutil==2.8.2",\
                        "PyWavelets==1.1.1",\
                        "scikit-image==0.17.2",\
                        "scipy==1.5",\
                        "six==1.16.0",\
                        "tifffile==2020.9.3",\
                        "scikit-image==0.17.2",\
                        "jupyter_server_proxy"]\
    )


