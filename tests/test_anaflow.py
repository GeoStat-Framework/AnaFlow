# -*- coding: utf-8 -*-
"""
This is the unittest of AnaFlow.
"""

import unittest
import numpy as np
import anaflow as af
# import matplotlib.pyplot as plt


def inc(arr, delta=0.0):
    return np.all(arr[:-1] <= arr[1:] + delta)


def dec(arr, delta=0.0):
    return np.all(arr[:-1] >= arr[1:] - delta)


def less(arr1, arr2, delta=0.0):
    return np.all(arr1 <= arr2 + delta)


class TestAnaFlow(unittest.TestCase):
    def setUp(self):
        self.delta = 1e-5
        self.t_gmean = 1e-4
        self.var = 0.5
        self.t_hmean = self.t_gmean * np.exp(-self.var / 2.0)
        self.anis = 0.4
        self.hurst = 0.7
        self.frac_dim = 2.3
        self.len_scale = 10.0
        self.lat_ext = 2.5
        self.r_ref = 20.0
        self.h_ref = -1.0
        self.rate = -1e-4
        self.storage = 1e-3
        self.rad = np.arange(1, 11, dtype=float)
        self.time = np.array([1e1, 1e2, 1e3, 1e4], dtype=float)

    def test_homogeneous(self):
        steady = af.thiem(
            rad=self.rad,
            r_ref=self.r_ref,
            transmissivity=self.t_gmean,
            rate=self.rate,
            h_ref=self.h_ref,
        )
        transient = af.theis(
            time=self.time,
            rad=self.rad,
            storage=self.storage,
            transmissivity=self.t_gmean,
            rate=self.rate,
            r_bound=self.r_ref,
            h_bound=self.h_ref,
        )
        self.assertTrue(inc(steady, self.delta))
        for rad_arr in transient:
            self.assertTrue(inc(rad_arr, self.delta))
        for time_arr in transient.T:
            self.assertTrue(dec(time_arr, self.delta))
        self.assertAlmostEqual(
            np.abs(np.max(steady - transient[-1, :])),
            0.0,
            places=4,
        )

    def test_ext_2d(self):
        steady = af.ext_thiem_2d(
            rad=self.rad,
            r_ref=self.r_ref,
            trans_gmean=self.t_gmean,
            var=self.var,
            len_scale=self.len_scale,
            rate=self.rate,
            h_ref=self.h_ref,
        )
        transient = af.ext_theis_2d(
            time=self.time,
            rad=self.rad,
            storage=self.storage,
            trans_gmean=self.t_gmean,
            var=self.var,
            len_scale=self.len_scale,
            rate=self.rate,
            r_bound=self.r_ref,
            h_bound=self.h_ref,
        )
        self.assertTrue(inc(steady, self.delta))
        for rad_arr in transient:
            self.assertTrue(inc(rad_arr, self.delta))
        for time_arr in transient.T:
            self.assertTrue(dec(time_arr, self.delta))
        self.assertAlmostEqual(
            np.abs(np.max(steady - transient[-1, :])),
            0.0,
            places=4,
        )

    def test_ext_3d(self):
        steady = af.ext_thiem_3d(
            rad=self.rad,
            r_ref=self.r_ref,
            cond_gmean=self.t_gmean,
            var=self.var,
            len_scale=self.len_scale,
            anis=self.anis,
            lat_ext=self.lat_ext,
            rate=self.rate,
            h_ref=self.h_ref,
        )
        transient = af.ext_theis_3d(
            time=self.time,
            rad=self.rad,
            storage=self.storage,
            cond_gmean=self.t_gmean,
            var=self.var,
            len_scale=self.len_scale,
            anis=self.anis,
            lat_ext=self.lat_ext,
            rate=self.rate,
            r_bound=self.r_ref,
            h_bound=self.h_ref,
        )
        self.assertTrue(inc(steady, self.delta))
        for rad_arr in transient:
            self.assertTrue(inc(rad_arr, self.delta))
        for time_arr in transient.T:
            self.assertTrue(dec(time_arr, self.delta))
        self.assertAlmostEqual(
            np.abs(np.max(steady - transient[-1, :])),
            0.0,
            places=4,
        )

    def test_neuman(self):
        steady = af.neuman2004_steady(
            rad=self.rad,
            r_ref=self.r_ref,
            trans_gmean=self.t_gmean,
            var=self.var,
            len_scale=self.len_scale,
            rate=self.rate,
            h_ref=self.h_ref,
        )
        transient = af.neuman2004(
            time=self.time,
            rad=self.rad,
            storage=self.storage,
            trans_gmean=self.t_gmean,
            var=self.var,
            len_scale=self.len_scale,
            rate=self.rate,
            r_bound=self.r_ref,
            h_bound=self.h_ref,
        )
        self.assertTrue(inc(steady, self.delta))
        for rad_arr in transient:
            self.assertTrue(inc(rad_arr, self.delta))
        for time_arr in transient.T:
            self.assertTrue(dec(time_arr, self.delta))
        self.assertAlmostEqual(
            np.abs(np.max(steady - transient[-1, :])),
            0.0,
            places=4,
        )

    def test_tpl(self):
        steady = af.ext_thiem_tpl(
            rad=self.rad,
            r_ref=self.r_ref,
            cond_gmean=self.t_gmean,
            var=self.var,
            len_scale=self.len_scale,
            lat_ext=self.lat_ext,
            hurst=self.hurst,
            dim=self.frac_dim,
            rate=self.rate,
            h_ref=self.h_ref,
        )
        transient = af.ext_theis_tpl(
            time=self.time,
            rad=self.rad,
            storage=self.storage,
            cond_gmean=self.t_gmean,
            var=self.var,
            len_scale=self.len_scale,
            lat_ext=self.lat_ext,
            hurst=self.hurst,
            dim=self.frac_dim,
            rate=self.rate,
            r_bound=self.r_ref,
            h_bound=self.h_ref,
        )
        self.assertTrue(inc(steady, self.delta))
        for rad_arr in transient:
            self.assertTrue(inc(rad_arr, self.delta))
        for time_arr in transient.T:
            self.assertTrue(dec(time_arr, self.delta))
        self.assertAlmostEqual(
            np.abs(np.max(steady - transient[-1, :])),
            0.0,
            places=3,
        )

    def test_tpl_3d(self):
        steady = af.ext_thiem_tpl_3d(
            rad=self.rad,
            r_ref=self.r_ref,
            cond_gmean=self.t_gmean,
            var=self.var,
            len_scale=self.len_scale,
            lat_ext=self.lat_ext,
            hurst=self.hurst,
            anis=self.anis,
            rate=self.rate,
            h_ref=self.h_ref,
        )
        transient = af.ext_theis_tpl_3d(
            time=self.time,
            rad=self.rad,
            storage=self.storage,
            cond_gmean=self.t_gmean,
            var=self.var,
            len_scale=self.len_scale,
            lat_ext=self.lat_ext,
            hurst=self.hurst,
            anis=self.anis,
            rate=self.rate,
            r_bound=self.r_ref,
            h_bound=self.h_ref,
        )
        self.assertTrue(inc(steady, self.delta))
        for rad_arr in transient:
            self.assertTrue(inc(rad_arr, self.delta))
        for time_arr in transient.T:
            self.assertTrue(dec(time_arr, self.delta))
        self.assertAlmostEqual(
            np.abs(np.max(steady - transient[-1, :])),
            0.0,
            places=2,
        )

        # plt.plot(self.rad, steady)
        # for rad_arr in transient:
        #     plt.plot(self.rad, rad_arr, dashes=[1, 1])
        # plt.show()


if __name__ == "__main__":
    unittest.main()
