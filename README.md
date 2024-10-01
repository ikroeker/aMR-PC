# aMR-PC
**Arbitrary Multi-Resolution Polynomial Chaos python toolbox**


This module can be installed by using `pip` inside this directory
```
pip install -e .
```
The `-e` switch means that pip will only link the source files to the directory
where all your python packages are installed, so that any changes in the source
code are taking place directly, and you don't have to reinstall after changes.
The dot command (.) is a  synonym for the current directory.

---

## Code sources

aMR-PC python code
[GitLab URL](https://git.iws.uni-stuttgart.de/ikroeker/ik_amr-pc)
and 
[Github URL](https://github.com/ikroeker/aMR-PC) 
Python implementation of arbirtrary multi-resolution polynomial chaos and multi-wavelets

---
## Package content

aMRPC: contains python package

tests: contains test input data in tests/data and several .py tests. 

---
## Related publications

 **Please cite the article:**

Ilja Kröker, Sergey Oladyshkin,
*Arbitrary multi-resolution multi-wavelet-based polynomial chaos expansion for data-driven uncertainty quantification.*
Reliability Engineering & System Safety,
Volume 222, 2022, 108376, ISSN 0951-8320,
https://doi.org/10.1016/j.ress.2022.108376.

Also used in following publications:

Rebecca Kohlhaas; Ilja Kröker; Sergey Oladyshkin; Wolfgang Nowak
*Gaussian active learning on multi-resolution arbitrary polynomial chaos emulator: concept for bias correction, assessment of surrogate reliability and its application to the carbon dioxide benchmark*
Comput Geosci (2023).
https://doi.org/10.1007/s10596-023-10199-1


Ilja Kroeker, Sergey Oladyshkin, Iryna Rybak
*Global sensitivity analysis using multi-resolution polynomial chaos expansion for coupled Stokes-Darcy flow problems.*
https://doi.org/10.1007/s10596-023-10236-z


Ilja Kröker, Tim Brünnette, Nils Wildt, Maria Fernanda Morales Oreamuno, Rebecca Kohlhaas, Sergey Oladyshkin, Wolfgang Nowak
*Bayesian³ Active learning for regularized arbitrary multi-element polynomial chaos using information theory*
https://doi.org/10.1615/int.j.uncertaintyquantification.2024052675
