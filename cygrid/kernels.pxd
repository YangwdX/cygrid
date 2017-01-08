#!python
# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: cdivision=True, boundscheck=False, wraparound=False
# cython: embedsignature=True

# ####################################################################
#
# title                  :kernels.pxd
# description            :Grid-kernel definitions.
# author                 :Benjamin Winkel, Lars Flöer & Daniel Lenz
#
# ####################################################################
#  Copyright (C) 2010+ by Benjamin Winkel, Lars Flöer & Daniel Lenz
#  bwinkel@mpifr.de, mail@lfloeer.de, dlenz.bonn@gmail.com
#  This file is part of cygrid.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# ####################################################################

from libc.math cimport cos, sin, sqrt, asin, fmod, exp, fabs
from numpy cimport (
    int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t,
    uint32_t, uint64_t, float32_t, float64_t
    )


cdef:
    # Kernels must have the following call signature:
    # double kernel(double distance, double bearing, void *kernel_params)
    # the latter parameter is assigned at runtime at must be one of the
    # following:

    public struct gaussian_1d_params:
        double inv_variance  # == 1. / 2. / sigma_kernel ** 2

    public struct gaussian_2d_params:
        double w_a
        double w_b
        double alpha

    public struct tapered_sinc_1d_params:
        double sigma
        double a  # should be set to 2.52
        double b  # should be set to 1.55

    public struct vector_1d_params:
        uint32_t n  # length of vector
        double refpix  # reference pixel number (zero-based!)
        # double refval  # assume, this is always zero! for speed
        double dx  # cell size (in degrees)
        double[::1] vector

    public struct matrix_2d_params:
        uint32_t n_x, n_y  # x/y size of matrix
        double refpix_x, refpix_y  # reference pixel number (zero-based!)
        # double refval_x, refval_y  # assume, this is always zero! for speed
        double dx, dy  # cell size (in degrees)
        double[::, ::1] matrix  # C-contiguos

    double gaussian_1d_kernel(
        double distance, double bearing, void *kernel_params
        ) nogil

    double gaussian_2d_kernel(
        double distance, double bearing, void *kernel_params
        ) nogil

    double tapered_sinc_1d_kernel(
        double distance, double bearing, void *kernel_params
        ) nogil

    double vector_1d_kernel(
        double distance, double bearing, void *kernel_params
        ) nogil

    double matrix_2d_kernel(
        double distance, double bearing, void *kernel_params
        ) nogil

