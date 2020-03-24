## Introduction
[PyHessian](https://github.com/amirgholami/PyHessian) is a pytorch library created by Yao, Gholami, Keutzer, and Mahoney of UCBerkeley and released alongside the paper, [PyHessian: Neural Networks Through the Lens of the Hessian](https://arxiv.org/pdf/1912.07145.pdf).

This repository is a fork of PyHessian and was created for the purposes of Stanford's Winter 2020 EE270 Course, taught by Prof. Mert Pilanci. While the course has completed, this repository is still a work in progress. Please feel free to read my [project report](https://github.com/iamsalil/PyHessian/blob/master/Project%20Final%20Report.pdf), which was the initial inspiration for this work.

## Change List
Here are a list of files that I made to PyHessian as a part of this project:
- pyhessian/hessian.py
- pyhessian/utils.py
- spectral_cdf.py (new)
- LanczosVisualizer.py (new)
- timer_pyhessian_standard.py (new)
- example_pyhessian_analysis.py (new)

## Findings
Here is a summary of my findings so far:
- The Lanczos Method is superior to PyHessian's Power Iteration for calculating top eigenvalues when calculating matvecs is costly
- Low dimensional sketches of the Hessian can give optimally accurate trace estimates and cheap, rough estimates of top eigenvalues (read the **Hessian Sketching** section of my report for the details of how this sketch is designed and what its properties are)

## Goals
1) Clean up my code, add comments, and make readable
2) Add functionality to create a partial CSD and use to compare top 100 eigenvalue distribution from sketched Hessian and original Hessian
3) Look at sketches of size 20/25/30/35
4) Explore matrix-free inverse iteration
