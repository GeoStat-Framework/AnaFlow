# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
"""
Cython-based acceleration kernels for :mod:`anaflow.flow.laplace`.

The Python wrapper takes care of all argument validation and prepares the
NumPy arrays that are passed into these routines.
"""

from libc.math cimport fabs, isinf, isnan, sqrt

import numpy as np

cimport numpy as np
from pentapy.solver cimport c_penta_solver2
from scipy.special.cython_special cimport iv as cy_iv
from scipy.special.cython_special cimport kv as cy_kv

ctypedef np.float64_t DTYPE_t
ctypedef np.intp_t ITYPE_t


cdef inline bint _should_zero(double value, double cut_off) nogil:
    """Return True if the value should be treated as zero."""
    if cut_off > 0.0:
        return fabs(value) < cut_off
    return value == 0.0


cdef inline void _shift_col_to_row(double[:, :] mat, Py_ssize_t width) nogil:
    """In-place conversion from column-wise to row-wise banded storage."""
    cdef Py_ssize_t up = 2
    cdef Py_ssize_t low = 2
    cdef Py_ssize_t i, j, shift, row

    for i in range(up):
        shift = up - i
        if shift >= width:
            for j in range(width):
                mat[i, j] = 0.0
            continue
        for j in range(width - shift):
            mat[i, j] = mat[i, j + shift]
        for j in range(width - shift, width):
            mat[i, j] = 0.0

    for i in range(low):
        shift = low - i
        row = 4 - i
        if shift >= width:
            for j in range(width):
                mat[row, j] = 0.0
            continue
        for j in range(width - 1, shift - 1, -1):
            mat[row, j] = mat[row, j - shift]
        for j in range(shift):
            mat[row, j] = 0.0


def solve_homogeneous(
    np.ndarray[DTYPE_t, ndim=1] s,
    np.ndarray[DTYPE_t, ndim=1] rad,
    np.ndarray[DTYPE_t, ndim=1] rad_pow,
    double r_well,
    double r_outer,
    double diff_sr0,
    double nu,
    double nu1,
    double gamma_1_minus_nu,
    double two_over_gamma_nu,
    np.ndarray[DTYPE_t, ndim=1] qs,
    double cut_off_prec,
    np.ndarray[DTYPE_t, ndim=2] out,
) -> None:
    """Homogeneous aquifer kernel."""
    cdef Py_ssize_t ns = s.shape[0]
    cdef Py_ssize_t nr = rad.shape[0]
    cdef Py_ssize_t si, ri
    cdef double se, cs, row00, row01, row10, row11
    cdef double rhs0, rhs1, det, inv_det, as_, bs
    cdef double rad_val, cs_rad, term_k, term_i, value
    cdef bint finite_outer = not isinf(r_outer)
    cdef double cut = cut_off_prec if cut_off_prec > 0.0 else 0.0

    for si in range(ns):
        se = s[si]
        cs = sqrt(se) * diff_sr0 if se >= 0.0 else 0.0
        row00 = -gamma_1_minus_nu
        row01 = two_over_gamma_nu
        row10 = 0.0
        row11 = 1.0

        if r_well > 0.0:
            value = cs * r_well
            row00 = -cy_kv(nu1, value)
            row01 = cy_iv(nu1, value)
        if finite_outer:
            value = cs * r_outer
            row10 = cy_kv(nu, value)
            row11 = cy_iv(nu, value)
        else:
            row01 = 0.0

        rhs0 = qs[si]
        rhs1 = 0.0
        det = row00 * row11 - row01 * row10
        if cut > 0.0 and fabs(det) < cut:
            as_ = 0.0
            bs = 0.0
        elif cut == 0.0 and det == 0.0:
            as_ = 0.0
            bs = 0.0
        else:
            inv_det = 1.0 / det
            as_ = (rhs0 * row11 - rhs1 * row01) * inv_det
            bs = (row00 * rhs1 - row10 * rhs0) * inv_det

        for ri in range(nr):
            rad_val = rad[ri]
            if finite_outer and not (rad_val < r_outer):
                out[si, ri] = 0.0
                continue

            cs_rad = cs * rad_val
            term_k = 0.0
            term_i = 0.0

            if cut > 0.0:
                if fabs(as_) >= cut:
                    term_k = as_ * cy_kv(nu, cs_rad)
                if fabs(bs) >= cut:
                    term_i = bs * cy_iv(nu, cs_rad)
            else:
                if as_ != 0.0:
                    term_k = as_ * cy_kv(nu, cs_rad)
                if bs != 0.0:
                    term_i = bs * cy_iv(nu, cs_rad)

            value = rad_pow[ri] * (term_k + term_i)
            if isnan(value) or isinf(value):
                value = 0.0
            out[si, ri] = value


def solve_multilayer(
    np.ndarray[DTYPE_t, ndim=1] s,
    np.ndarray[DTYPE_t, ndim=1] rad,
    np.ndarray[DTYPE_t, ndim=1] rad_pow,
    np.ndarray[DTYPE_t, ndim=1] r_part,
    np.ndarray[DTYPE_t, ndim=1] difsr,
    np.ndarray[DTYPE_t, ndim=1] tmp,
    np.ndarray[ITYPE_t, ndim=1] pos,
    double nu,
    double nu1,
    double gamma_1_minus_nu,
    double two_over_gamma_nu,
    np.ndarray[DTYPE_t, ndim=1] qs,
    double cut_off_prec,
    np.ndarray[DTYPE_t, ndim=2] out,
) -> None:
    """General multi-layer kernel."""
    cdef Py_ssize_t ns = s.shape[0]
    cdef Py_ssize_t nr = rad.shape[0]
    cdef Py_ssize_t parts = difsr.shape[0]
    cdef Py_ssize_t width_total = 2 * parts
    cdef Py_ssize_t si, ri, i, row, col, first, width, idx, p
    cdef double se, val, cs_i, cs_ip1, tmp_i, r_interface
    cdef double cs_rad, term_k, term_i, value
    cdef double row00, row01, row10, row11, rhs0, rhs1, det, inv_det
    cdef double inv_cutoff = 0.0
    cdef double cut = cut_off_prec if cut_off_prec > 0.0 else 0.0
    cdef double even_val, odd_val
    cdef bint finite_outer = np.isfinite(r_part[r_part.shape[0] - 1])

    if cut_off_prec > 0.0:
        inv_cutoff = 1.0 / cut_off_prec

    mb_np = np.zeros((5, width_total), dtype=np.float64)
    rhs_np = np.zeros(width_total, dtype=np.float64)
    x_np = np.zeros(width_total, dtype=np.float64)
    cs_np = np.zeros(parts, dtype=np.float64)
    col_max_np = np.zeros(width_total, dtype=np.float64)
    work_np = np.zeros((5, width_total), dtype=np.float64)

    cdef double[:, :] mb = mb_np
    cdef double[:] rhs = rhs_np
    cdef double[:] x = x_np
    cdef double[:] cs = cs_np
    cdef double[:] col_max = col_max_np
    cdef double[:, :] work = work_np

    for si in range(ns):
        se = s[si]
        if se >= 0.0:
            for i in range(parts):
                cs[i] = sqrt(se) * difsr[i]
        else:
            for i in range(parts):
                cs[i] = 0.0

        for row in range(5):
            for col in range(width_total):
                mb[row, col] = 0.0

        for idx in range(width_total):
            rhs[idx] = 0.0
            x[idx] = 0.0

        rhs[0] = qs[si]
        mb[2, 0] = -gamma_1_minus_nu
        mb[1, 1] = two_over_gamma_nu
        mb[2, width_total - 1] = 1.0

        for i in range(parts - 1):
            r_interface = r_part[i + 1]
            cs_i = cs[i] * r_interface
            cs_ip1 = cs[i + 1] * r_interface
            tmp_i = tmp[i]

            mb[0, 2 * i + 3] = -cy_iv(nu, cs_ip1)
            mb[1, 2 * i + 2] = -cy_kv(nu, cs_ip1)
            mb[1, 2 * i + 3] = -cy_iv(nu1, cs_ip1)
            mb[2, 2 * i + 1] = cy_iv(nu, cs_i)
            mb[2, 2 * i + 2] = cy_kv(nu1, cs_ip1)
            mb[3, 2 * i] = cy_kv(nu, cs_i)
            mb[3, 2 * i + 1] = tmp_i * cy_iv(nu1, cs_i)
            mb[4, 2 * i] = -tmp_i * cy_kv(nu1, cs_i)

        if r_part[0] > 0.0:
            val = cs[0] * r_part[0]
            mb[2, 0] = -cy_kv(nu1, val)
            mb[1, 1] = cy_iv(nu1, val)
        if finite_outer:
            val = cs[parts - 1] * r_part[r_part.shape[0] - 1]
            mb[3, width_total - 2] = cy_kv(nu, val)
            mb[2, width_total - 1] = cy_iv(nu, val)
        else:
            mb[0, width_total - 1] = 0.0
            mb[1, width_total - 1] = 0.0

        first = parts
        for col in range(width_total):
            value = 0.0
            for row in range(5):
                val = fabs(mb[row, col])
                if val > value:
                    value = val
            col_max[col] = value
            if cut > 0.0:
                if value < cut or (inv_cutoff > 0.0 and value > inv_cutoff):
                    first = col // 2
                    break
            else:
                if value == 0.0:
                    first = col // 2
                    break

        if first > parts:
            first = parts

        for idx in range(2 * first, width_total):
            x[idx] = 0.0

        if first <= 1:
            row00 = mb[2, 0]
            row01 = mb[1, 1]
            row10 = mb[3, 0]
            row11 = mb[2, 1]
            rhs0 = rhs[0]
            rhs1 = rhs[1]
            det = row00 * row11 - row01 * row10
            if cut > 0.0 and fabs(det) < cut:
                x[0] = 0.0
                x[1] = 0.0
            elif cut == 0.0 and det == 0.0:
                x[0] = 0.0
                x[1] = 0.0
            else:
                inv_det = 1.0 / det
                x[0] = (rhs0 * row11 - rhs1 * row01) * inv_det
                x[1] = (row00 * rhs1 - row10 * rhs0) * inv_det
        else:
            width = 2 * first
            for row in range(5):
                for col in range(width):
                    work[row, col] = mb[row, col]
            if first < parts:
                work[4, width - 1] = 0.0
                work[3, width - 1] = 0.0
                work[4, width - 2] = 0.0
            _shift_col_to_row(work, width)
            try:
                sol = c_penta_solver2(work[:, :width], rhs[:width])
                for idx in range(width):
                    x[idx] = sol[idx]
            except ZeroDivisionError:
                for idx in range(width):
                    x[idx] = 0.0

        for idx in range(2 * first):
            if _should_zero(x[idx], cut):
                x[idx] = 0.0

        for ri in range(nr):
            p = pos[ri]
            cs_rad = cs[p] * rad[ri]
            term_k = 0.0
            term_i = 0.0
            even_val = x[2 * p]
            odd_val = x[2 * p + 1]

            if cut > 0.0:
                if fabs(even_val) >= cut:
                    term_k = even_val * cy_kv(nu, cs_rad)
                if fabs(odd_val) >= cut:
                    term_i = odd_val * cy_iv(nu, cs_rad)
            else:
                if even_val != 0.0:
                    term_k = even_val * cy_kv(nu, cs_rad)
                if odd_val != 0.0:
                    term_i = odd_val * cy_iv(nu, cs_rad)

            value = rad_pow[ri] * (term_k + term_i)
            if isnan(value) or isinf(value):
                value = 0.0
            out[si, ri] = value
