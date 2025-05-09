from setuptools import setup


setup(\
        name="yuca", \
        packages=["yuca"], \
        version = "0.0", \
        description = "Your Universal Cellular Automata", \
        install_requires=[\
                        "bokeh==2.4.1",\
                        "coverage==7.0.3",\
                        "numpy==1.22.0",\
                        "torch==2.0.1",\
                        "future==0.18.3",\
                        "pyparsing==3.0.7",\
                        "python-dateutil==2.8.2",\
                        "scipy==1.10.0",\
                        "six==1.16.0",\
                        "tifffile==2020.9.3"]\
    )


