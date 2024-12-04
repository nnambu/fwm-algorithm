# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:26:42 2024

@author: Noa
"""

""" Test the delay of adding fused silica in the code
"""

import numpy as np
import matplotlib.pyplot as plt
from pypret import material

import pypret
from pypret import (FourierTransform, Pulse, random_pulse, random_gaussian, PNPS, MeshDataPlot, MeshData, lib)


def FS_ngroup(x, unit='om'):
    if (unit=='wl'):
        x = lib.twopi*lib.sol/x
    FS_sellmeier_coeffs = [0.6961663, 0.0684043,
                       0.4079426, 0.1162414,
                       0.8974794, 9.8961610]
    nphase = material.FS.n(x, unit='om')
    dndw = x*0
    for i in range(0, 3):
        Ai = FS_sellmeier_coeffs[2*i]
        Bi = FS_sellmeier_coeffs[2*i+1]/(lib.twopi*lib.sol*1e6)
        dndw += x/nphase*Ai*Bi**2/(1 - (x*Bi)**2)**2
    return nphase + x*dndw



ft = FourierTransform(512, dt=5e-15)
pulse = Pulse(ft, 800e-9)
w0 = pulse.w0
n = material.FS.n(ft.w + w0, unit='om')

# print(FS_ngroup(800e-9, unit='wl'))


n0 =  material.FS.n(w0, unit='om')

# plt.plot(lib.twopi*lib.sol/(ft.w + w0), n)
# plt.plot(ft.w, n)
# plt.plot(ft.w, n0 + ft.w*(1.4671-n0)/pulse.w0)
# plt.plot(ft.w, n0 + ft.w*(FS_ngroup(w0, unit='om')-n0)/pulse.w0)