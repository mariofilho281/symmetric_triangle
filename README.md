## Computational appendix to *[Local models and Bell inequalities for the minimal triangle network](https://www.arxiv.org/abs/????.?????)*
#### José Mário da Silva, Alejandro Pozas-Kerstjens, and Fernando Parisio 

This repository contains the computational appendix to the paper "*Local models and Bell inequalities for the minimal triangle network*". It provides the codes that were used to generate the figures in the manuscript, and several utility functions, such as tests to ascertain which kind of nonlocality a given behavior exhibits, functions to calculate the inequalities reported in the paper, and routines to generate distributions on the proposed local boundaries.

Most of the code is written in Python, but there are also Mathematica notebooks showing how to obtain the analytical results of the paper.

Libraries required:

- [numpy](https://www.numpy.org) for vectorized array operations.
- [scipy](https://scipy.org/) for solving optimization problems.
- [matplotlib](https://matplotlib.org) for 2D plots.
- [pyvista](https://pyvista.org/) for 3D plots.
- [inflation](https://www.github.com/ecboghiu/inflation) (and its 
  requirements) for setting up and solving the compatibility problems.
- [tqdm](https://tqdm.github.io/) for progress bars.

Files:

* Figure generation:
  - [figure_symmetric_local_set.py](https://github.com/mariofilho281/symmetric_triangle/blob/main/figure_symmetric_local_set.py): generates an interactive version of the 3D representation of the proposed local boundaries.
  - [fig4a.py](https://github.com/mariofilho281/symmetric_triangle/blob/main/fig4a.py) and [fig4b.py](https://github.com/mariofilho281/symmetric_triangle/blob/main/fig4b.py): generate 2D plots showing the regions accessible with local strategies and the NSI-incompatible behaviors.

* Utilities
  - [triangle_inequalities.py](https://github.com/mariofilho281/symmetric_triangle/blob/main/triangle_inequalities.py): functions for evaluating the inequalities, test nonlocality and membership in the GHZ and W regions, and generating points on the local boundaries.
  - [exploratory_analysis.py](https://github.com/mariofilho281/symmetric_triangle/blob/main/exploratory_analysis.py): script to explore local models in a user-defined affine subspace (change equation in line 29).
  - [boundary_validation.py](https://github.com/mariofilho281/symmetric_triangle/blob/main/boundary_validation.py): script used to validate proposed local boundaries.
  - [triangle_inequalities.py](https://github.com/mariofilho281/symmetric_triangle/blob/main/triangle.py): class for local model fitting.
  - [polynomials.py](https://github.com/mariofilho281/symmetric_triangle/blob/main/polynomials.py): functions used to calculate the W cyan surface.

* Mathematica notebooks
  - [GHZ.nb](https://github.com/mariofilho281/symmetric_triangle/blob/main/GHZ.nb): model and inequality calculations for the GHZ surface.
  - [W_green.nb](https://github.com/mariofilho281/symmetric_triangle/blob/main/W_green.nb): model and inequality calculations for the W green surface.
  - [W_purple.nb](https://github.com/mariofilho281/symmetric_triangle/blob/main/W_purple.nb): model and inequality calculations for the W purple surface.
  - [W_red.nb](https://github.com/mariofilho281/symmetric_triangle/blob/main/W_red.nb): model and inequality calculations for the W red surface.
  - [W_yellow.nb](https://github.com/mariofilho281/symmetric_triangle/blob/main/W_yellow.nb): model and inequality calculations for the W yellow surface.
  - [W_cyan.nb](https://github.com/mariofilho281/symmetric_triangle/blob/main/W_cyan.nb): model and inequality calculations for the W cyan surface.

#### Citing
If you would like to cite this work, please use the following format:

J. M. da Silva, A. Pozas-Kerstjens, and F. Parisio, _Local models and Bell inequalities for the minimal triangle network_, ???, arXiv:????.?????

```
@article{minimal_triangle,
  title = {Local models and Bell inequalities for the minimal triangle network},
  author = {da Silva, Jos\'e M\'ario and Pozas-Kerstjens, Alejandro and 
  Parisio, Fernando},
  journal = {???},
  volume = {???},
  pages = {???},
  doi = {???},
  url = {???},
  archivePrefix = {arXiv},
  eprint = {???},
  year = {2025}
}
```
