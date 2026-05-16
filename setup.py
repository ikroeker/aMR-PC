import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aMRPC",
    version="1.0.5",
    author="Ilja Kröker",
    author_email="ilja.kroeker@iws.uni-stuttgart.de",
    description="aMRPC python implementation",
    keywords="arbirtray multi-resolution polynomial chaos, multi-wavelet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ikroeker/aMR-PC",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        "numpy", "pandas", "scipy",  "numba"
        ],
    project_urls={
        'LS3':'https://www.iws.uni-stuttgart.de/ls3/',
        'IWS':'https://www.iws.uni-stuttgart.de'
    },
)
