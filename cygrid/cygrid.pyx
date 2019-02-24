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


__all__ = ['Cygrid', 'WcsGrid', 'SlGrid', 'ShapeError']


from .kernels cimport (
    gaussian_1d_kernel, gaussian_2d_kernel, tapered_sinc_1d_kernel
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
ctypedef double (*kernel_func_ptr_t)(double, double, double[::1]) nogil


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
    Fast Cython-powered gridding software, base class.

    DO NOT USE DIRECTLY, BUT THE DERIVED CLASSES `~cygrid.WcsGrid` OR
    `~cygrid.SlGrid`.

    The underlying algorithm is a based on serialized convolution with finite
    gridding kernels. Currently, only Gaussian kernels are provided (which has
    the drawback of slight degradation of the effective resolution). The
    algorithm has very small memory footprint, allows easy parallelization,
    and is very fast.

    Look into the `~cygrid.Cygrid.set_kernel` and `~cygrid.Cygrid.grid`
    methods help for more information on how to use this.

    Internally, we make use of the HEALPix representation for book-keeping.
    The idea is the following: for each input point we query which HEALPix
    pixels are located within the required convolution kernel radius (using
    HEALPix `query_disc` function). Likewise, for the target map pixels (any
    WCS projection) we calculate the HEALPix index they live in. By a simple
    cross- matching (hash-map based) we can thus easily find out which input
    pixels contribute to which output pixels. In practice it is a little more
    complicated, because world pixels could share the same HEALPix index. We
    use lists (or rather C++ vectors) to account for this.

    Parameters
    ----------
    Optional keyword arguments:
        dbg_messages : Boolean (default: False)
            Do debugging output
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
        bint bearing_needed, kernel_set
        kernel_func_ptr_t kernel_func_ptr
        np.ndarray kernel_params_arr

        # helper lookup tables for faster processing are wrapped in
        #  the HpxHashTable class; see associated docs for more information
        HpxHashTable my_hpx_hashtab
        bint dbg_messages

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

        This is a convenience function, to call `omp_set_num_threads()`,
        which is otherwise not possible during runtime from Python.
        '''

        openmp.omp_set_num_threads(nthreads)

    def _prepare(self, *args, **kwargs):
        '''
        Preparation function, called by derived Classes.

        Needs to fill/prepare the following class members:

        # 2D representation
        np.ndarray xpix, ypix, xwcs, ywcs
        # flat version (1D) of above, must not contain NaNs!
        np.ndarray xpix_f, ypix_f, xwcs_f, ywcs_f
        # shapes, to allow sanity checks
        tuple zyx_shape, yx_shape

        '''

        raise NotImplementedError('This is the base class. Use child classes!')

    def clear_cache(self):
        '''
        Clear all internal caches/dictionaries.

        Notes
        -----

        To speed-up processing, `~cygrid` uses internal caches (see `Cygrid
        paper <http://adsabs.harvard.edu/abs/2016A%26A...591A..12W>`_ for
        details). This is usually a minor contribution to the total memory
        usage, but there could be scenarios, e.g., when one grids into very
        large maps having very small pixel sizes, that the internal cache
        grows too much. In this case, we recommend to sort the input data by
        latitude, grid the data in chunks, and call this function every now
        and then.'''

        self.my_hpx_hashtab.clear_hashes

    def clear_data_and_weights(self):
        '''
        Reset data and weights arrays. (Caches stay filled!)
        '''

        self.datacube[...] = 0.
        self.weightscube[...] = 0.

    def set_kernel(
            self,
            object kernel_type, object kernel_params,
            double sphere_radius, double hpx_max_resolution
            ):
        '''
        Set the gridding kernel type and parameters.

        Parameters
        ----------
        kernel_type : str
            Set the kernel type.

            The following names/types are available:
            'gauss1d', 'gauss2d', 'tapered_sinc' (see Notes for details)
        kernel_params : `~numpy.array`
            Set the kernel parameters for the chosen type (see Notes for
            details)
        sphere_radius : double
            Kernel sphere radius.

            This is controls out to which distance the kernel
            is computed for. For Gaussian kernels, values much larger
            than :math:`3\\ldots4 \\sigma_\\mathrm{k}` do not make much sense.
        hpx_max_resolution : double
            Maximum acceptable HPX resolution
            (:math:`\\sigma_\\mathrm{k}^{[\\mathrm{maj}]} / 2` is a
            reasonable value).

        Notes
        -----
        Below you find a list of kernel-names and required parameters::

            'gauss1d', (kernel_sigma,)
            'gauss2d', (kernel_sigma_maj, kernel_sigma_min, PA)
            'tapered_sinc', (kernel_sigma, param_a, param_b)

        All numbers are in units of degrees, except for `PA`, `param_a` and
        `param_b`. `PA` (the position angle) is in units of radians (for
        efficiency). `param_a` and `param_b` should be `2.52` and `1.55`,
        respectively, for optimal results!

        The kernel size (sigma) defines the amount of "smoothness"
        applied to the data. If in doubt a good value is about 50%
        of the true/input angular resolution of the data (this will result
        in about 10% degradation of the final angular resolution.)
        '''

        cdef:
            unicode kernel_type_u = ustring(kernel_type)
            unicode kernel_description
            # double (*kernel_func_ptr)(double, double, double[::1]) nogil

            np.ndarray kernel_params_arr = np.atleast_1d(
                kernel_params
                ).astype(np.float64)
            double[::1] kparams_v = kernel_params_arr  # test if mem-view works

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

        if kparams_v.shape[0] != num_params:
            raise ValueError('kernel_params needs {} entries for {}'.format(
                num_params, kernel_type_u
                ))

        if kernel_type_u == 'gauss1d':
            self.kernel_func_ptr = gaussian_1d_kernel
            kparams_v[0] = 0.5 / kparams_v[0] ** 2
        elif kernel_type_u == 'gauss2d':
            self.kernel_func_ptr = gaussian_2d_kernel
        elif kernel_type_u == 'tapered_sinc':
            self.kernel_func_ptr = tapered_sinc_1d_kernel

        self.kernel_set = <bint> True
        self.kernel_params_arr = kernel_params_arr

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
        Grid irregularly positioned data points (spectra) into the data cube.

        After successful gridding, you can obtain the resulting datacube with
        the `~cygrid.Cygrid.get_datacube` method. The associated weight cube
        is accessible with `~cygrid.Cygrid.get_weights`.

        Parameters
        ----------
        lons, lats : `~numpy.array` [1D] of float64
            Flat lists/arrays of input coordinates :math:`(l, b)`.
        data/weights : `~numpy.array` [1D or 2D] of float32 or float64
            The spectra and their weights (optional) for each of the given
            coordinate pairs, :math:`(l, b)`. First axis must match lons/lats size. Internally, `~cygrid` always works with a (possibly redundant) spectral axis. Thus, the gridded data returned by
            `~cygrid.Cygrid.get_datacube` is always a 3D array. However,
            to simplify the interface, this method will add a
            spectral dimension for you, if you provide a 1D-array, only.
        dtype : str ['float32' or 'float64']
            Desired output format of data cube. (Default: 'float32').

        Raises
        ------
        ShapeError
            Input coordinates/data points length mismatch.
            Number of spectral channels mismatch.

        Notes
        -----
        - All input parameters need to be C-contiguous (`~cygrid` will
          re-cast if necessary).
        '''

        if weights is None:
            weights = np.ones_like(data)

        if not self.kernel_set:
            raise RuntimeError('No kernel has been set, use set_kernel method')

        if lons.ndim != 1 or lats.ndim != 1:
            raise ShapeError('Input coordinates must be 1D objects.')

        if data.ndim != weights.ndim:
            raise ShapeError('Data and weights arrays must have same shape.')

        if data.ndim not in [1, 2]:
            raise ShapeError('Input data/weights must be 1D or 2D objects.')

        if data.ndim == 1:
            data = data[:, np.newaxis]
            weights = weights[:, np.newaxis]

        if not (
                len(lons) == len(lats) == len(data) == len(weights)
                ):
            raise ShapeError('Input coordinates/data points length mismatch.')

        if not (
                len(data[0]) == len(weights[0]) == self.zyx_shape[0]
                ):
            raise ShapeError('Number of spectral channels mismatch.')

        arr_kw_32 = dict(dtype=FLOAT32, requirements='C')
        arr_kw_64 = dict(dtype=FLOAT64, requirements='C')
        if dtype == 'float32':

            if self.dbg_messages:
                print('User requested single precision.')

            self.datacube = self.datacube.astype(FLOAT32, copy=False)
            self.weightscube = self.weightscube.astype(FLOAT32, copy=False)
            data = np.require(data, **arr_kw_32)
            weights = np.require(weights, **arr_kw_32)

        elif dtype == 'float64':

            if self.dbg_messages:
                print('User requested double precision.')

            self.datacube = self.datacube.astype(FLOAT64, copy=False)
            self.weightscube = self.weightscube.astype(FLOAT64, copy=False)
            data = np.require(data, **arr_kw_64)
            weights = np.require(weights, **arr_kw_64)

        else:
            raise TypeError("dtype must be one of 'float32' or 'float64'")

        lons = np.require(lons, **arr_kw_64)
        lats = np.require(lats, **arr_kw_64)

        self._grid(lons, lats, data, weights)

    def _grid(
            self,
            double[::1] lons not None,
            double[::1] lats not None,
            cython.floating[:, ::1] data not None,
            cython.floating[:, ::1] weights not None,
            ):


        cdef:
            int64_t i  # Windows open-mp only works with signed loop vars
            uint64_t z, y, x, j, k
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
            int64_t outlen
            uint64_t in_idx
            uint64_t _pix, _totpix, _goodpix

            double l1, l2, b1, b2, sinbdiff, sinldiff
            double sdist, sbear, sweight, tweight

            # make local copies for faster lookup
            double disc_size = self.disc_size
            double sphere_radius = self.sphere_radius
            bint bearing_needed = self.bearing_needed
            kernel_func_ptr_t kernel_func_ptr = self.kernel_func_ptr
            double[::1] kernel_params_v = self.kernel_params_arr

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
                        sdist, sbear, kernel_params_v
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
        '''
        Return final data.

        Returns
        -------
        data : `~numpy.array` [2D or 3D] of float32 or float64
            The gridded spectral data.
        '''
        return self.datacube / self.weightscube

    def get_weights(self):
        '''
        Return final weights.

        Returns
        -------
        weights : `~numpy.array` [2D or 3D] of float32 or float64
            The gridded spectral weights.
        '''
        return self.weightscube

    def get_unweighted_datacube(self):
        '''
        Return final unweighted data. (For debugging only.)

        Returns
        -------
        unweighted_data : `~numpy.array` [2D or 3D] of float32 or float64
            The gridded spectral unweighted data.
        '''
        return self.datacube


cdef class WcsGrid(Cygrid):
    '''
    WCS version of `~cygrid.Cygrid`.

    Parameters
    ----------
    header : Dictionary or anything that fits into `~astropy.wcs.WCS`
        The header must contain a valid `~astropy.wcs.WCS` representation for
        a three-dimensional data cube (spatial-spatial-frequency).
    dbg_messages : Boolean, optional (Default: False)
        Do debugging output.
    datacube : `~numpy.array` [3D] of float32 or float64, optional (Default: None)
        Provide pre-allocated or even pre-filled `~numpy.array` for datacube.

        Usually (if `datacube=None`, the default) a datacube object will
        be created automatically according to the given FITS header
        dictionary. Providing datacube manually might be worthwhile if some
        kind of repeated/iterative gridding process is desired. However, for
        almost all use cases it will be sufficient to repeatedly call the
        grid method. Cygrid won't clear the memory itself initially, so make
        sure to handle this correctly.
    weightcube : `~numpy.array` [3D] of float32 or float64, optional (Default: None)
        As `datacube` but for the weight array.

    Raises
    ------
    TypeError
        Input datacube/weightcube must be numpy array.
        Input datacube must have floating point type.
        Weightcube dtype doesn't match datacube dtype.
    ShapeError
        Datacube/weightcube shape doesn't match fits header.

    Examples
    --------
    The following provides a minimal example. For more detailed information
    we refer to the user manual or the Jupyter tutorial notebooks::

        from astropy.io import fits
        import matplotlib.pyplot as plt
        import cygrid

        # read-in data
        lon, lat, rawdata = get_data()

        # define target FITS/WCS header
        header = create_fits_header()

        # prepare gridder
        kernelsize_sigma = 0.2

        kernel_type = 'gauss1d'
        kernel_params = (kernelsize_sigma, )  # must be a tuple
        kernel_support = 3 * kernelsize_sigma
        hpx_maxres = kernelsize_sigma / 2

        mygridder = cygrid.WcsGrid(header)
        mygridder.set_kernel(
            kernel_type,
            kernel_params,
            kernel_support,
            hpx_maxres,
            )

        # do the actual gridding
        mygridder.grid(lon, lat, rawdata)

        # query result and store to disk ...
        data_cube = mygridder.get_datacube()
        fits.writeto('example.fits', header=header, data=data_cube)

        # ... or plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=WCS(header).celestial)
        ax.imshow(
            data_cube[0], origin='lower', interpolation='nearest'
            )
        plt.show()

    '''

    cdef:
        object header
        object wcs
        np.ndarray coordmask_f

    def _prepare(self, header, **kwargs):

        # if naxis3 kwarg is set, it always takes precedence; if it is not
        # set, NAXIS3 from header is used; if it is not even present in
        # header, set to One
        naxis3 = kwargs.get('naxis3', None)
        if naxis3 is None:
            naxis3 = header.get('NAXIS3', 1)

        self.header = header
        self.zyx_shape = (naxis3, header['NAXIS2'], header['NAXIS1'])
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
            self.weightscube = np.zeros(self.zyx_shape, dtype=np.float32)

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
        '''
        Return WCS object for reference.

        Returns
        -------
        wcs : `~astropy.wcs.WCS`
            WCS object created from input header dictionary.
        '''
        return self.wcs

    def get_world_coords(self):
        '''
        Return world coordinates of the datacube's xy-plane

        Returns
        -------
        lons/lats : `~numpy.array` [2D]
            World coordinates :math:`(l, b)` of the datacube.
        '''
        return self.xwcs, self.ywcs

    def get_pixel_coords(self):
        '''
        Return pixel coordinates of the datacube's xy-plane

        Returns
        -------
        x/y : `~numpy.array` [2D]
            Pixel coordinates :math:`(x, y)` of the datacube.
        '''
        return self.xpix, self.ypix

    def get_header(self):
        '''
        Return header object for reference.

        Returns
        -------
        header : dict or `~astropy.wcs.WCS`-compatible
            Header object provided to the constructor.
        '''
        return self.header

    def get_datacube(self):
        '''
        Return final data cube.

        Returns
        -------
        data : `~numpy.array` [3D] of float32 or float64
            The gridded data cube as defined by WCS header.
        '''
        return self.datacube / self.weightscube

    def get_weights(self):
        '''
        Return final weights.

        Returns
        -------
        weights : `~numpy.array` [3D] of float32 or float64
            The gridded weights cube as defined by WCS header.
        '''
        return self.weightscube

    def get_unweighted_datacube(self):
        '''
        Return final unweighted data. (For debugging only.)

        Returns
        -------
        unweighted_data : `~numpy.array` [3D] of float32 or float64
            The gridded unweighted data cube as defined by WCS header.
        '''
        return self.datacube


cdef class SlGrid(Cygrid):
    '''
    Sight line version of `~cygrid.Cygrid`.

    The sight line gridder can be used to resample input data to any collection
    of output coordinates. For example, one could grid to the (list of)
    HEALPix grid pixel coordinates, or just extract spectra from a large 3D
    survey on selected positions (not necessarily aligned with the pixels).

    Parameters
    ----------
    sl_lons, sl_lats : numpy.ndarray (float64), 1D
        Coordinates of sight lines to grid onto.
    naxes3 : int (Default: 1)
        Length of spectral axis.
    dbg_messages : Boolean, optional (Default: False)
        Do debugging output.
    datacube : `~numpy.array` [2D] of float32 or float64, optional (Default: None)
        Provide pre-allocated or even pre-filled `~numpy.array` for datacube.

        Usually (if `datacube=None`, the default) a datacube object will
        be created automatically according to the given FITS header
        dictionary. Providing datacube manually might be worthwhile if some
        kind of repeated/iterative gridding process is desired. However, for
        almost all use cases it will be sufficient to repeatedly call the
        grid method. Cygrid won't clear the memory itself initially, so make
        sure to handle this correctly.

    weightcube : `~numpy.array` [2D] of float32 or float64, optional (Default: None)
        As `datacube` but for the weight array.

    Raises
    ------
    TypeError
        Input datacube/weightcube must be numpy array.
        Input datacube must have floating point type.
        Weightcube dtype doesn't match datacube dtype.
    ShapeError
        Datacube/weightcube shape doesn't match sight line dimensions.

    Examples
    --------
    The following provides a minimal example. For more detailed information
    we refer to the user manual or the Jupyter tutorial notebooks::

        from astropy.io import fits
        import matplotlib.pyplot as plt
        import cygrid

        # read-in data, lon/lat are 1D, input_signal has 2nd dimension: 1,
        # i.e., we are not gridding spectra but single values
        input_lon, input_lat, input_signal = get_data()

        # prepare gridder
        kernelsize_sigma = 0.2

        kernel_type = 'gauss1d'
        kernel_params = (kernelsize_sigma, )  # must be a tuple
        kernel_support = 3 * kernelsize_sigma
        hpx_maxres = kernelsize_sigma / 2

        mygridder = cygrid.SlGrid(target_lon, target_lat, 1)
        mygridder.set_kernel(
            kernel_type,
            kernel_params,
            kernel_support,
            hpx_maxres,
            )

        # do the actual gridding
        mygridder.grid(input_lon, input_lat, input_signal)
        target_signal = gridder.get_datacube()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(target_lon, target_lat, c=target_signal[:, 0])
        plt.show()

    '''

    cdef:
        object header
        object wcs
        np.ndarray coordmask_f

    def _prepare(
            self,
            np.ndarray[double, ndim=1] sl_lons,
            np.ndarray[double, ndim=1] sl_lats,
            long naxes3=1,
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
            self.weightscube = np.zeros(self.zyx_shape, dtype=np.float32)

        self.ypix, self.xpix = np.indices(self.yx_shape, dtype=UINT64)
        self.xwcs_f, self.ywcs_f = sl_lons, sl_lats
        self.xwcs, self.ywcs = (
            self.xwcs_f.reshape(self.yx_shape),
            self.ywcs_f.reshape(self.yx_shape)
            )

        # keep flat versions for later use
        self.ypix_f, self.xpix_f = self.ypix.flatten(), self.xpix.flatten()

    def get_datacube(self):
        '''
        Return final data.

        Returns
        -------
        data : `~numpy.array` [2D] of float32 or float64
            The gridded spectral data for each of the input positions.
        '''
        return self.datacube[:, 0] / self.weightscube[:, 0]

    def get_weights(self):
        '''
        Return final weights.

        Returns
        -------
        weights : `~numpy.array` [2D] of float32 or float64
            The gridded spectral weights for each of the input positions.
        '''
        return self.weightscube[:, 0]

    def get_unweighted_datacube(self):
        '''
        Return final unweighted data. (For debugging only.)

        Returns
        -------
        unweighted_data : `~numpy.array` [2D] of float32 or float64
            The gridded spectral unweighted data for each of the input positions.
        '''
        return self.datacube[:, 0]
