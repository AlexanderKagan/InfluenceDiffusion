from setuptools import setup, find_packages

<<<<<<< HEAD
VERSION = '0.0.11'
=======
VERSION = '0.0.9'
>>>>>>> parent of b603939 (Update setup.py)
DESCRIPTION = 'InfluenceDiffusion package'
LONG_DESCRIPTION = 'in this package, we implement popular network diffusion models and methods for their estimation.'

setup(
    name="InfluenceDiffusion",
    version=VERSION,
    author="Alexander Kagan",
    author_email="<amkagan@umich.edu>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=["numpy",
                      "scipy",
                      "networkx",
                      "typing",
                      "joblib"],

    keywords=['python', 'Influence Maximization', "Network diffusion models",
              "General Linear Threshold model", "Social Networks", "Independent Cascade model"],

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
