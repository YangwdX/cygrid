#!python
# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: cdivision=True, boundscheck=False, wraparound=False
# cython: embedsignature=True

# ####################################################################
#
# title                  :cygrid.pyx
# description            :Grid data points to map or sightlines.
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
# Note: Some HEALPix related helper functions are adapted from the
#   official Healpix_cxx (HEALPix C++) library.
#   (Copyright (C) 2003-2012 Max-Planck-Society; author Martin Reinecke)
#   This was denoted in the docstrings accordingly.
#   For more information about HEALPix, see http://healpix.sourceforge.net
#   Healpix_cxx is being developed at the Max-Planck-Institut fuer Astrophysik
#   and financially supported by the Deutsches Zentrum fuer Luft- und Raumfahrt
#   (DLR).
# ####################################################################

# import python3 compat modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import std lib
import sys
import traceback

# import cython specifics
cimport cython
from cython.parallel import prange
from cython.operator cimport dereference as deref, preincrement as inc
from cpython cimport bool as python_bool
cimport openmp

# import C/C++ modules
from libc.math cimport exp, cos, sin, sqrt, asin, acos, atan2, fabs, fmod
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.set cimport set as cpp_set
from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map

# import numpy/data types
import numpy as np
from numpy cimport (
    int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t,
    uint32_t, uint64_t, float32_t, float64_t
    )
cimport numpy as np

# import WCS support lib
from astropy import wcs


np.import_array()

INT8 = np.dtype(np.int8)
INT16 = np.dtype(np.int16)
INT32 = np.dtype(np.int32)
INT64 = np.dtype(np.int64)
UINT8 = np.dtype(np.uint8)
UINT16 = np.dtype(np.uint16)
UINT32 = np.dtype(np.uint32)
UINT64 = np.dtype(np.uint64)
FLOAT32 = np.dtype(np.float32)
FLOAT64 = np.dtype(np.float64)


__all__ = ['WcsGrid', 'SlGrid']


from .kernels cimport (
    gaussian_1d_kernel, gaussian_1d_params,
    gaussian_2d_kernel, gaussian_2d_params,
    tapered_sinc_1d_kernel, tapered_sinc_1d_params,
    vector_1d_kernel, vector_1d_params,
    matrix_2d_kernel, matrix_2d_params,
    )
from .hphashtab cimport HpxHashTable
from .helpers cimport (
    ustring, ilog2, isqrt, fmodulo, nside_to_order,
    npix_to_nside, true_angular_distance, great_circle_bearing
    )
from .constants cimport NESTED, RING, MAX_Y
from .constants cimport DEG2RAD, RAD2DEG
from .constants cimport PI, TWOTHIRD, TWOPI, HALFPI, INV_TWOPI, INV_HALFPI


# Define maximal y-size of pixel indices
#  this is necessary to have a quick'n'dirty hash for the xpix-ypix pairs
#  otherwise we would need to provide a hash function to unordered_map,
DEF MAX_Y = 2**30

# define function pointers (1D and 2D), to allow user-chosen kernels
# double distance, double bearing, double[::1] kernel_params)
# (use bearing=NULL for 1D kernels)
ctypedef double (*kernel_func_ptr_t)(double, double, void *kernel_params) nogil


# define some helper functions
def eprint():
    '''
    Print python exception and backtrace.
    '''

    print(sys.exc_info()[1])
    print(traceback.print_tb(sys.exc_info()[2]))


# define some custom exception classes
class ShapeError(Exception):
    '''
    This exception is for mismatches of WCS header and actual data cube
    sizes or input data sizes.
    '''


cdef class Cygrid(object):
    '''
    Fast cython-powered gridding software. Base class.

    The method is a based on serialized convolution with finite gridding
     kernels. Currently, only Gaussian kernels are provided (which has the
     drawback of slight degradation of the effective resolution). The algorithm
     has very small memory footprint, allows easy parallelization, and is very
     fast.

    Look into the "grid" method help for more information on how to use this.

    Internally, we make use of the HEALPix representation for book-keeping.
     The idea is the following: for each input point we query which HPX pixels
     are located within the required convolution kernel radius (using
     HEALPix query_disc function). Likewise, for the target map pixels
     (any WCS projection) we calculate the HPX index they live in. By a simple
     cross-matching (hash-map based) we can thus easily find out which input
     pixels contribute to which output pixels. In practice it is a little more
     complicated, because world pixels could share the same HPX index. We
     use lists (or rather C++ vectors) to account for this.

    Parameters
    ----------
    Optional keyword arguments:
        dbg_messages: do debugging output
    '''

    cdef:
        # datacube and weights objects, any or all dimensions can have length 1
        np.ndarray datacube, weightscube

        # pixel/world coords (2D) of internal datacube spatial dims
        # if astropy is used, the coords may contain NaNs
        # in that case, one has to calculate the (shortened) versions of
        # the flat lists, otherwise, one can just copy
        # 2D representation
        np.ndarray xpix, ypix, xwcs, ywcs
        # flat version (1D) of above, must not contain NaNs!
        np.ndarray xpix_f, ypix_f, xwcs_f, ywcs_f
        # shapes, to allow sanity checks
        tuple zyx_shape, yx_shape

        uint64_t nside
        double disc_size
        double sphere_radius
        double last_sphere_radius, last_hpxmaxres, hpx_resol
        # np.ndarray kernel_params_arr

        # helper lookup tables for faster processing are wrapped in
        #  the HpxHashTable class; see associated docs for more information
        HpxHashTable my_hpx_hashtab
        bint dbg_messages

        bint bearing_needed, kernel_set
        kernel_func_ptr_t kernel_func_ptr
        # need to instantiate each kernel type, even if not needed
        # (cannot dynamically do it)
        gaussian_1d_params _gaussian_1d_params
        gaussian_2d_params _gaussian_2d_params
        tapered_sinc_1d_params _tapered_sinc_1d_params
        vector_1d_params _vector_1d_params
        matrix_2d_params _matrix_2d_params
        void *kernel_params_pointer  # points to correct kernel_params struct

    def __init__(self, *args, **kwargs):
        # Constructor will initalize necessary cube/weights arrays, setup cube
        # and wcs representation

        self.dbg_messages = <bint> False
        if 'dbg_messages' in kwargs:
            self.dbg_messages = kwargs['dbg_messages']

        self.my_hpx_hashtab = HpxHashTable(dbg_messages=self.dbg_messages)

        self.last_sphere_radius = -1.
        self.last_hpxmaxres = -1.
        # self.bearing_needed = <bint> False
        self.kernel_set = <bint> False

        self._prepare(*args, **kwargs)

    def set_num_threads(self, int nthreads):
        '''
        Change maximum number of threads to use.

        This is a convenience function, to call omp_set_num_threads(),
        which is not possible during runtime from python.
        '''

        openmp.omp_set_num_threads(nthreads)

    def _prepare(self, *args, **kwargs):
        '''
        Preparation function, called by derived classes (the implementations).

        Needs to fill/prepare the following class members:

        # 2D representation
        np.ndarray xpix, ypix, xwcs, ywcs
        # flat version (1D) of above, must not contain NaNs!
        np.ndarray xpix_f, ypix_f, xwcs_f, ywcs_f
        # shapes, to allow sanity checks
        tuple zyx_shape, yx_shape

        '''

        raise NotImplementedError('This is the base class. Use child classes!')

    def set_kernel(
            self,
            object kernel_type, object kernel_params,
            double sphere_radius, double hpx_max_resolution
            ):
        '''
        Set the gridding kernel type and parameters.

        Parameters
        ----------
        kernel_type : string type (python3: unicode, python2: bytes)
            set the kernel type, the following names/types are available:
            'gauss1d', 'gauss2d', 'tapered_sinc' (see Notes for details)
        kernel_params : tuple
            set the kernel parameters for the chosen type (see Notes for
            details)
        sphere_radius : double
            Kernel sphere radius.
            This is controls out to which distance the kernel
            is computed for. For Gaussian kernels, values much larger
            than 3 (sigma) do not make much sense.
        hpx_max_resolution : double
            Maximum acceptable HPX resolution (kernel_sigma[_maj] / 2.
            is a reasonable value).

        Notes
        -----
        Below you find a list of kernel-names and required parameters:
            'gauss1d', (kernel_sigma,)
            'gauss2d', (kernel_sigma_maj, kernel_sigma_min, PA)
            'tapered_sinc', (kernel_sigma, param_a, param_b)

        Except for param_a and param_b all numbers are in units of degrees.
        Param_a and Param_b should be 2.52 and 1.55, respectively, for optimal
        results!

        The kernel size (sigma) defines the amount of "smoothness"
        applied to the data. If in doubt a good value is about 25%
        of the true/input angular resolution of the data (this will result
        in about 10% degradation of the final angular resolution.)
        '''

        cdef:
            unicode kernel_type_u = ustring(kernel_type)
            unicode kernel_description
            # double (*kernel_func_ptr)(double, double, double[::1]) nogil

            # np.ndarray kernel_params_arr = np.atleast_1d(
            #     kernel_params
            #     ).astype(np.float64)
            # double[::1] kernel_params = kernel_params_arr  # test if mem-view works

            int num_params

            dict kernel_types = {
                'gauss1d': (
                    '1D gaussian kernel', 1, <bint> False,
                    # ('0.5 / kernel_sigma ** 2',)
                    ),
                'gauss2d': (
                    '2D gaussian kernel', 3, <bint> True,
                    # ('kernel_sigma_maj', 'kernel_sigma_min', 'PA')
                    ),
                'tapered_sinc': (
                    '1D tapered-sinc kernel', 3, <bint> False,
                    # ('kernel_sigma', 'param_a', 'param_b')
                    ),
                'vector1d': (
                    '1D vector discrete kernel', 3, <bint> False,
                    # ('kernel vector', 'refpix', 'dx')
                    ),
                'matrix2d': (
                    '2D matrix discrete kernel', 3, <bint> True,
                    # ('kernel matrix', ('refpix_x', 'refpix_y), ('dx', 'dy'))
                    ),
                }

        try:
            (
                kernel_description, num_params, self.bearing_needed
                ) = kernel_types[kernel_type_u]

            if self.dbg_messages:
                print('Using {}'.format(kernel_description))
                # print('# parametes {}'.format(num_params))
                print('Kernel type {}'.format(kernel_type_u))
                print('Kernel Parameters {}'.format(kernel_params))

        except KeyError:
            raise TypeError(
                'Kernel type not understood: {}\n'.format(kernel_type_u) +
                'Please choose from the following\n' +
                '\n'.join(kernel_types.keys())
                )

        if len(kernel_params) != num_params:
            raise ValueError('kernel_params needs {} entries for {}'.format(
                num_params, kernel_type_u
                ))

        if kernel_type_u == 'gauss1d':

            self.kernel_func_ptr = gaussian_1d_kernel
            self._gaussian_1d_params.inv_variance = 0.5 / kernel_params[0] ** 2
            self.kernel_params_pointer = &(self._gaussian_1d_params)

        elif kernel_type_u == 'gauss2d':

            self.kernel_func_ptr = gaussian_2d_kernel
            self._gaussian_2d_params.w_a = <double> kernel_params[0]
            self._gaussian_2d_params.w_b = <double> kernel_params[1]
            self._gaussian_2d_params.alpha = (
                (<double> kernel_params[3]) * PI / 180.
                )
            self.kernel_params_pointer = &(self._gaussian_2d_params)

        elif kernel_type_u == 'tapered_sinc':

            self.kernel_func_ptr = tapered_sinc_1d_kernel
            self._tapered_sinc_1d_params.sigma = <double> kernel_params[0]
            self._tapered_sinc_1d_params.a = <double> kernel_params[1]
            self._tapered_sinc_1d_params.b = <double> kernel_params[2]
            self.kernel_params_pointer = &(self._tapered_sinc_1d_params)

        elif kernel_type_u == 'vector1d':

            self.kernel_func_ptr = vector_1d_kernel
            self._vector_1d_params.vector = np.ascontiguousarray(
                kernel_params[0]
                )
            self._vector_1d_params.refpix = <double> kernel_params[1]
            self._vector_1d_params.dx = <double> kernel_params[2]
            self._vector_1d_params.n = <uint32_t> len(kernel_params[0])
            self.kernel_params_pointer = &(self._vector_1d_params)

        elif kernel_type_u == 'matrix2d':

            self.kernel_func_ptr = matrix_2d_kernel
            self._matrix_2d_params.matrix = np.ascontiguousarray(
                kernel_params[0]
                )
            self._matrix_2d_params.refpix_x = <double> kernel_params[1][0]
            self._matrix_2d_params.refpix_y = <double> kernel_params[1][1]
            self._matrix_2d_params.dx = <double> kernel_params[2][0]
            self._matrix_2d_params.dy = <double> kernel_params[2][1]
            self._matrix_2d_params.n_x = <uint32_t> len(kernel_params[0][0])
            self._matrix_2d_params.n_y = <uint32_t> len(kernel_params[0])
            self.kernel_params_pointer = &(self._matrix_2d_params)

        self.kernel_set = <bint> True
        # self.kernel_params_arr = kernel_params_arr

        # recompute hpx lookup table in case kernel sphere has changed
        # if you want to use very different kernels, you should call the grid
        # function with the same kernels sizes subsequently before changing
        # to the next kernel size
        # Note: 3e-5 rad is about 0.1 arcsec
        if (
                abs(self.last_sphere_radius - sphere_radius) > 3e-5 or
                abs(self.last_hpxmaxres - hpx_max_resolution) > 3e-5  # or
                ):
            # prepare_helpers needs only to be called, if the internal
            # hpx representation needs a change, e.g., if necessary resolution
            # has changed or if sphere radius is different (need empty cache)
            self.my_hpx_hashtab.prepare_helpers(
                hpx_max_resolution,
                self.xpix_f,
                self.ypix_f,
                self.xwcs_f,
                self.ywcs_f,
                )   # this will also wipe disks cache
            self.last_sphere_radius = sphere_radius
            self.last_hpxmaxres = hpx_max_resolution

        self.sphere_radius = sphere_radius
        self.disc_size = (
            DEG2RAD * sphere_radius + self.my_hpx_hashtab.resolution
            )
        if self.dbg_messages:
            print(
                'Disc size: {:.4f} arcmin'.format(
                    RAD2DEG * self.disc_size * 60.
                ))

    # This is just a thin wrapper around _grid to allow default-arg handling
    # and streamlining/sanity checking the inputs
    def grid(
            self,
            np.ndarray lons, np.ndarray lats,
            np.ndarray data, np.ndarray weights=None,
            dtype='float32',
            ):
        '''
        Grid irregular data points (spectra) into the data cubes.

        After successful gridding, you can obtain the resulting datacube with
        the "get_datacube method".

        Parameters
        ----------
        lons, lats : numpy.ndarray (float64), 1D
            Flat lists of coordinates.
        data/weights : numpy.ndarray (float32 or float64), 2D
            The spectra and their weights for each of the given coordinate
            pairs (lons, lats). First axis must match lons/lats size.
            Second axis is the spectral dimension.
        dtype : string-like
            Output format of data cube, either 'float32' (default) or 'float64'

        Raises
        ------
        ShapeError
            Input coordinates/data points length mismatch.
            Number of spectral channels mismatch.

        Notes
        -----
        - All input parameters need to C-contiguous
          (use numpy.ascontiguousarray to recast if necessary)
        - It is possible to produce maps instead of datacubes by using
          spectra of length 1.
        '''

        if weights is None:
            weights = np.ones_like(data)

        if not self.kernel_set:
            raise RuntimeError('No kernel has been set, use set_kernel method')

        if lons.ndim != 1 or lats.ndim != 1:
            raise ShapeError('Input coordinates must be 1D objects.')

        if data.ndim != 2 or weights.ndim != 2:
            raise ShapeError('Input data/weights must be 2D objects.')

        if not (
                len(lons) == len(lats) == len(data) == len(weights)
                ):
            raise ShapeError('Input coordinates/data points length mismatch.')

        if not (
                len(data[0]) == len(weights[0]) == self.zyx_shape[0]
                ):
            raise ShapeError('Number of spectral channels mismatch.')

        d32 = np.array([1.], dtype=np.float32)
        d64 = np.array([1.], dtype=np.float64)
        if dtype == 'float32':

            if self.dbg_messages:
                print('User requested single precision.')

            self.datacube = self.datacube.astype(d32.dtype.str, copy=False)
            self.weightscube = self.weightscube.astype(d32.dtype.str, copy=False)
            data = data.astype(d32.dtype.str, copy=False)
            weights = weights.astype(d32.dtype.str, copy=False)
        elif dtype == 'float64':

            if self.dbg_messages:
                print('User requested double precision.')

            self.datacube = self.datacube.astype(d64.dtype.str, copy=False)
            self.weightscube = self.weightscube.astype(d64.dtype.str, copy=False)
            data = data.astype(d64.dtype.str, copy=False)
            weights = weights.astype(d64.dtype.str, copy=False)
        else:
            raise TypeError("dtype must be one of 'float32' or 'float64'")

        lons = lons.astype(d64.dtype.str, copy=False)
        lats = lats.astype(d64.dtype.str, copy=False)

        self._grid(lons, lats, data, weights)

    def _grid(
            self,
            double[::1] lons not None,
            double[::1] lats not None,
            cython.floating[:, :] data not None,
            cython.floating[:, :] weights not None,
            ):


        cdef:
            uint64_t z, y, x, i, j, k
            uint64_t speccount = len(data)
            uint64_t numchans = len(data[0])

            # create (local) views of the ndarrays for faster access
            cython.floating[:, :, :] datacubeview = self.datacube
            cython.floating[:, :, :] weightscubeview = self.weightscube
            double[:, :] ywcsview = self.ywcs
            double[:, :] xwcsview = self.xwcs

            # lookup-tables
            # this is necessary to massively parallelize the grid routine
            # if one would just go through the list of input coords,
            # one could have race conditions (during write access) because
            # multiple input positions could contribute to the same target
            # pixel
            # building the lookup-tables is wrapped in the HpxHashTable
            # helper class
            unordered_map[uint64_t, vector[uint64_t]] output_input_mapping
            vector[uint64_t] output_pixels
            vector[uint64_t] input_pixels
            uint64_t outlen, in_idx
            uint64_t _pix, _totpix, _goodpix

            double l1, l2, b1, b2, sinbdiff, sinldiff
            double sdist, sbear, sweight, tweight

            # make local copies for faster lookup
            double disc_size = self.disc_size
            double sphere_radius = self.sphere_radius
            bint bearing_needed = self.bearing_needed
            kernel_func_ptr_t kernel_func_ptr = self.kernel_func_ptr
            # double[::1] kernel_params_v = self.kernel_params_arr
            void *kernel_params_pointer = self.kernel_params_pointer

        if self.dbg_messages:
            print('Gridding {} spectra in datacube...'.format(len(data)))

        # calculate_output_pixels must be called everytime new input
        # coordinates are to be processed
        self.my_hpx_hashtab.calculate_output_pixels(
            lons,
            lats,
            disc_size,
            output_input_mapping,  # this is modified
            output_pixels,  # this is modified
            )

        outlen = output_pixels.size()
        _totpix = 0
        _goodpix = 0

        for i in prange(outlen, nogil=True, schedule='guided', chunksize=100):

            _pix = output_pixels[i]
            x = _pix // MAX_Y
            y = _pix % MAX_Y
            l1 = DEG2RAD * xwcsview[y, x]
            b1 = DEG2RAD * ywcsview[y, x]

            input_pixels = output_input_mapping[output_pixels[i]]
            _totpix += input_pixels.size()

            for j in range(input_pixels.size()):
                in_idx = input_pixels[j]

                l2 = DEG2RAD * lons[in_idx]
                b2 = DEG2RAD * lats[in_idx]

                sdist = RAD2DEG * true_angular_distance(l1, b1, l2, b2)
                if bearing_needed:
                    sbear = great_circle_bearing(l1, b1, l2, b2)  # rad

                if sdist < sphere_radius:
                    _goodpix += 1
                    sweight = kernel_func_ptr(
                        sdist, sbear, kernel_params_pointer
                        )
                    for z in range(numchans):
                        tweight = weights[in_idx, z] * sweight
                        datacubeview[z, y, x] += data[in_idx, z] * tweight
                        weightscubeview[z, y, x] += tweight

        if self.dbg_messages:
            print('# of target pixels used: {}'.format(outlen))
            print(
                '# of input-output pixel combinations: {}'.format(_totpix)
                )
            print('# of good input-output pixel combinations: {}'.format(
                _goodpix
                ))
            print(
                'Avg. # of input pixels / output pixel: {:.1f}'.format(
                    float(_totpix) / float(outlen)
                ))
            print(
                'Avg. # of good input pixels / output pixel: {:.1f}'.format(
                    float(_goodpix) / float(outlen)
                ))

    def get_datacube(self):
        '''Return final data cube.'''
        return self.datacube / self.weightscube

    def get_weights(self):
        '''Return final weights cube.'''
        return self.weightscube

    def get_unweighted_datacube(self):
        '''Return unweighted data cube.'''
        return self.datacube


cdef class WcsGrid(Cygrid):
    '''
    WCS-grid-version of cygrid.

    Parameters
    ----------
    header : Dictionary or anything that fits into astropy.wcs.WCS
        The header must contain a valid wcs representation for a 3-dimensional
        data cube (spatial-spatial-frequency).

    Optional keyword arguments:
        dbg_messages: do debugging output
        datacube : floating-point numpy.ndarray
            Provide pre-allocated numpy array for datacube.
            Usually (if datacube=None, the default) a datacube object will
            be created automatically according to fits header dictionary.
            Providing datacube manually might be worthwhile if some kind of
            repeated/iterative gridding process is desired. However, for
            almost all use cases it will be sufficient to repeatedly call the
            grid method.

            Cygrid won't clear the memory itself initially, so make sure to
            handle this correctly.

        weightcube : floating-point numpy.ndarray
            As datacube but for the weight array.

    Raises
    ------
    TypeError
        Input datacube/weightcube must be numpy array.
        Input datacube must have floating point type.
        Weightcube dtype doesn't match datacube dtype.
    ShapeError
        Datacube/weightcube shape doesn't match fits header.
    '''

    cdef:
        object header
        object wcs
        np.ndarray coordmask_f

    def _prepare(self, header, **kwargs):

        self.header = header
        self.zyx_shape = (header['NAXIS3'], header['NAXIS2'], header['NAXIS1'])
        self.yx_shape = self.zyx_shape[1:3]

        # Use astropy's wcs module to convert pixel <--> world coords
        #  need celestial part of coordinates only
        try:
            self.wcs = wcs.WCS(self.header, naxis=[wcs.WCSSUB_CELESTIAL])
        except TypeError:
            # some astropy versions have a bug
            self.wcs = wcs.WCS(self.header).sub(axes=[1, 2])

        if 'datacube' in kwargs:
            self.datacube = kwargs['datacube']
            if not isinstance(self.datacube, np.ndarray):
                raise TypeError('Input datacube must be numpy array.')
            if not (
                    self.datacube.shape[0] == self.zyx_shape[0] and
                    self.datacube.shape[1] == self.zyx_shape[1] and
                    self.datacube.shape[2] == self.zyx_shape[2]
                    ):
                raise ShapeError("Datacube shape doesn't match fits header.")
            if self.datacube.dtype not in [
                    np.float32, np.float64
                    ]:
                raise TypeError(
                    'Input datacube must have floating point type.'
                    )
        else:
            self.datacube = np.zeros(self.zyx_shape, dtype=np.float32)

        if 'weightcube' in kwargs:
            self.weightscube = kwargs['weightcube']
            if not isinstance(self.weightscube, np.ndarray):
                raise TypeError('Input weightcube must be numpy array.')
            if not (
                    self.datacube.shape[0] == self.weightscube.shape[0] and
                    self.datacube.shape[1] == self.weightscube.shape[1] and
                    self.datacube.shape[2] == self.weightscube.shape[2]
                    ):
                raise ShapeError(
                    "Weightcube shape doesn't match datacube shape."
                    )
            if self.datacube.dtype != self.weightscube.dtype:
                raise TypeError(
                    "Weightcube dtype doesn't match datacube dtype."
                    )
        else:
            self.weightscube = self.datacube.copy()

        self.ypix, self.xpix = np.indices(self.yx_shape, dtype=UINT64)

        # keep flat versions for later use
        self.ypix_f, self.xpix_f = self.ypix.flatten(), self.xpix.flatten()

        # calculate associated world coordinates
        self.xwcs_f, self.ywcs_f = self.wcs.wcs_pix2world(
            self.xpix_f + 1, self.ypix_f + 1, 1
            )
        self.xwcs = self.xwcs_f.reshape(self.yx_shape).astype(np.float64)
        self.ywcs = self.ywcs_f.reshape(self.yx_shape).astype(np.float64)

        # astropy puts invalid coords to NaN, need a validation mask
        self.coordmask_f = np.isfinite(self.xwcs_f)
        self.xwcs_f = self.xwcs_f[self.coordmask_f]
        self.ywcs_f = self.ywcs_f[self.coordmask_f]
        self.xpix_f = self.xpix_f[self.coordmask_f]
        self.ypix_f = self.ypix_f[self.coordmask_f]

        if self.dbg_messages:
            print('Target field edge coordinates:')
            print(
                '-> left lon: {:.6f} right lon {:.6f}\n'
                '-> top lat: {:.6f} bottom lat {:.6f}'.format(
                    self.xwcs[self.xwcs.shape[0] / 2, 0],
                    self.xwcs[self.xwcs.shape[0] / 2, -1],
                    self.ywcs[0, self.xwcs.shape[1] / 2],
                    self.ywcs[-1, self.xwcs.shape[1] / 2],
                    )
                )

    def get_wcs(self):
        '''Return WCS object for reference.'''
        return self.wcs

    def get_world_coords(self):
        '''Return world coordinates of cube's xy-plane'''
        return self.xwcs, self.ywcs

    def get_pixel_coords(self):
        '''Return pixel coordinates of cube's xy-plane'''
        return self.xpix, self.ypix

    def get_header(self):
        '''Return header object for reference.'''
        return self.header


cdef class SlGrid(Cygrid):
    '''
    Sight line-grid version of cygrid.

    The sight line gridder can be used to resample input data to any collection
    of output coordinates. For example, one could grid to the (list of)
    HEALPix grid pixel coordinates, or just extract spectra from a large 3D
    survey on selected positions (not necessarily aligned with the pixels).

    Parameters
    ----------
    sl_lons, sl_lats : numpy.ndarray (float64), 1D
        Coordinates of sight lines to grid onto.
    naxes3 : int
        Length of spectral axis.

    Optional keyword arguments:
        dbg_messages: do debugging output
        datacube : floating-point numpy.ndarray
            Provide pre-allocated numpy array for datacube.
            Usually (if datacube=None, the default) a datacube object will
            be created automatically according to fits header dictionary.
            Providing datacube manually might be worthwhile if some kind of
            repeated/iterative gridding process is desired. However, for
            almost all use cases it will be sufficient to repeatedly call the
            grid method.

            Cygrid won't clear the memory itself initially, so make sure to
            handle this correctly.

        weightcube : floating-point numpy.ndarray
            As datacube but for the weight array.

    Raises
    ------
    TypeError
        Input datacube/weightcube must be numpy array.
        Input datacube must have floating point type.
        Weightcube dtype doesn't match datacube dtype.
    ShapeError
        Datacube/weightcube shape doesn't match sight line dimensions.
    '''

    cdef:
        object header
        object wcs
        np.ndarray coordmask_f

    def _prepare(
            self,
            np.ndarray[double, ndim=1] sl_lons,
            np.ndarray[double, ndim=1] sl_lats,
            long naxes3,
            **kwargs
            ):

        self.zyx_shape = (naxes3, 1, len(sl_lons))
        self.yx_shape = self.zyx_shape[1:3]

        if 'datacube' in kwargs:
            self.datacube = kwargs['datacube']
            if not isinstance(self.datacube, np.ndarray):
                raise TypeError('Input datacube must be numpy array.')
            if not (
                    self.datacube.shape[0] == self.zyx_shape[0] and
                    self.datacube.shape[1] == self.zyx_shape[1] and
                    self.datacube.shape[2] == self.zyx_shape[2]
                    ):
                raise ShapeError(
                    "Datacube shape doesn't match sight line dimensions."
                    )
            if self.datacube.dtype not in [
                    np.float32, np.float64
                    ]:
                raise TypeError(
                    'Input datacube must have floating point type.'
                    )
        else:
            self.datacube = np.zeros(self.zyx_shape, dtype=np.float32)

        if 'weightcube' in kwargs:
            self.weightscube = kwargs['weightcube']
            if not isinstance(self.weightscube, np.ndarray):
                raise TypeError('Input weightcube must be numpy array.')
            if not (
                    self.datacube.shape[0] == self.weightscube.shape[0] and
                    self.datacube.shape[1] == self.weightscube.shape[1] and
                    self.datacube.shape[2] == self.weightscube.shape[2]
                    ):
                raise ShapeError(
                    "Weightcube shape doesn't match sight line dimensions."
                    )
            if self.datacube.dtype != self.weightscube.dtype:
                raise TypeError(
                    "Weightcube dtype doesn't match datacube dtype."
                    )
        else:
            self.weightscube = self.datacube.copy()

        self.ypix, self.xpix = np.indices(self.yx_shape, dtype=UINT64)
        self.xwcs_f, self.ywcs_f = sl_lons, sl_lats
        self.xwcs, self.ywcs = (
            self.xwcs_f.reshape(self.yx_shape),
            self.ywcs_f.reshape(self.yx_shape)
            )

        # keep flat versions for later use
        self.ypix_f, self.xpix_f = self.ypix.flatten(), self.xpix.flatten()
