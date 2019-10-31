import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aMRPC-pkg",
    version="0.0.1",
    author="Ilja Kröker",
    author_email="ilja.kroeker@iws.uni-stuttgart.de",
    description="aMRPC python implementation",
    keywords="arbirtray multi-resolution polynomial chaos, multi-wavelet",
    long_description=long_description,
    #long_description_content_type="text/markdown",
    url="https://git.iws.uni-stuttgart.de/ikroeker/ik_amr-pc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    requires=[
        "numpy", "pandas", "scipy", 
        ],
    project_urls={
        'LS3':'https://www.iws.uni-stuttgart.de/ls3/',
        'IWS':'https://www.iws.uni-stuttgart.de'
    },
)
