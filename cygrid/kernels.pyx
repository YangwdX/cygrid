#!python
# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: cdivision=True, boundscheck=False, wraparound=False
# cython: embedsignature=True

# ####################################################################
#
# title                  :kernels.pyx
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

from .constants cimport PI


__all__ = []


cdef double sinc(double x) nogil:
    '''
    Sinc function with simple singularity check.
    '''

    if fabs(x) < 1.e-20:
        return 1.
    else:
        return sin(x) / x


cdef double gaussian_1d_kernel(
        double distance, double bearing, void *kernel_params
        ) nogil:
    '''
    Gaussian-1D kernel function.

    Parameters
    ----------
    distance : double
        Radial distance/separation
    bearing : double
        unused - this is only in the call signature to allow function pointers
    kernel_params : gaussian_1d_params struct
        see "kernels.pxd" for description

    Returns
    -------
    Kernel weight : double
    '''

    cdef gaussian_1d_params *params = <gaussian_1d_params*> kernel_params

    return exp(-distance * distance * params.inv_variance)


cdef double gaussian_2d_kernel(
        double distance, double bearing, void *kernel_params
        ) nogil:
    '''
    Gaussian-2D kernel function.

    Parameters
    ----------
    distance : double
        Radial distance/separation
    bearing : double
        Bearing of a position w.r.t. kernel center
    kernel_params : gaussian_2d_params struct
        see "kernels.pxd" for description


    Returns
    -------
    Kernel weight : double
    '''

    cdef:
        gaussian_2d_params *params = <gaussian_2d_params*> kernel_params
        double ellarg, Earg

    ellarg = (
        params.w_a ** 2 * sin(bearing - params.alpha) ** 2 +
        params.w_b ** 2 * cos(bearing - params.alpha) ** 2
        )
    Earg = (distance / params.w_a / params.w_b) ** 2 / 2. * ellarg

    return exp(-Earg)


cdef double tapered_sinc_1d_kernel(
        double distance, double bearing, void *kernel_params
        ) nogil:
    '''
    Kaiser-Bessel-1D kernel function (Gaussian-tapered sinc).

    Parameters
    ----------
    distance : double
        Radial distance/separation
    bearing : double
        unused - this is only in the call signature to allow function pointers
    kernel_params : tapered_sinc_1d_params struct
        see "kernels.pxd" for description

    Returns
    -------
    Kernel weight : double

    Notes: the sigma_kernel widths is defined in a manner compatible with
        gaussian_1d, i.e., the kernel-sphere radius should also be three times
        as large. See
            http://casa.nrao.edu/aips2_docs/glossary/g.html
        for details.
    '''

    cdef:
        tapered_sinc_1d_params *params = \
            <tapered_sinc_1d_params*> kernel_params
        double arg

    arg = PI * distance / params.sigma

    return sinc(arg / params.b) * exp(-(arg / params.a) ** 2)


cdef double vector_1d_kernel(
        double distance, double bearing, void *kernel_params
        ) nogil:
    '''
    Radial-vector 1D kernel function.

    Parameters
    ----------
    distance : double
        Radial distance/separation
    bearing : double
        unused - this is only in the call signature to allow function pointers
    kernel_params : vector_1d_params struct
        see "kernels.pxd" for description

    Returns
    -------
    Kernel weight : double
    '''

    cdef:
        vector_1d_params *params = <vector_1d_params*> kernel_params
        uint32_t index

    index = <uint32_t> (
        # assume, refval_x is always zero! for speed
        # (x - refval) / dx + refpix + 0.5
        distance / params.dx + params.refpix + 0.5
        )

    if index < 0 or index >= params.n:
        return 0.

    return params.vector[index]


cdef double matrix_2d_kernel(
        double distance, double bearing, void *kernel_params
        ) nogil:
    '''
    Matrix-2D kernel function.

    Parameters
    ----------
    distance : double
        Radial distance/separation
    bearing : double
        Bearing of a position w.r.t. kernel center
    kernel_params : matrix_2d_params struct
        see "kernels.pxd" for description

    Returns
    -------
    Kernel weight : double
    '''

    cdef:
        matrix_2d_params *params = <matrix_2d_params*> kernel_params
        uint32_t index_x, index_y
        double x, y

    # bearing is counted from North, but we want matrix
    # kernels that have origin at lower left
    x = distance * sin(bearing)
    y = -distance * cos(bearing)

    index_x = <uint32_t> (
        # assume, refval_x is always zero! for speed
        # (x - refval_x) / dx + refpix_x + 0.5
        x / params.dx + params.refpix_x + 0.5
        )
    index_y = <uint32_t> (
        # assume, refval_y is always zero! for speed
        # (y - refval_y) / dy + refpix_y + 0.5
        y / params.dy + params.refpix_y + 0.5
        )

    if (
            index_x < 0 or index_y < 0 or
            index_x >= params.n_x or index_y >= params.n_y
            ):
        return 0.

    return params.matrix[index_y, index_x]
