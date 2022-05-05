from setuptools import setup


setup(\
        name="yuca", \
        packages=["yuca"], \
        version = "0.0", \
        description = "Your Universal Cellular Automata", \
        install_requires=[\
                        "bokeh==2.4.1",\
                        "jupyter==1.0.0",\
                        "notebook==6.3.0",\
                        "numpy==1.18.4",\
                        "torch==1.5.1",\
                        "scikit-image==0.17.2",\
                        "jupyter_server_proxy",\
                        "matplotlib==3.3.3"]\
    )


