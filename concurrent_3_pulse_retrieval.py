# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 09:31:10 2024

Do two concurrent retrievals to retrieve 800, 400, and 266 nm pulses
Plot a comparison figure of all 3

@author: Noa
"""



import pypret
from pypret.frequencies import convert
from pypret import (random_gaussian, random_pulse, Retriever, MeshDataPlot, lib)
import numpy as np
import matplotlib.pyplot as plt
import create_frog_input
import fwm_xfrog_pnps
import glob
from test_result_plots import (plot_field_comparison, plot_retrieved_field, plot_trace_comparison)
import test_result_plots2
import multi_xfrog_retriever

grating = "5000"
FS_beam = 400

nscans = 2

# wavelengths of the unknown pulses (either 400 and 800, or 400 and 266)
wl400 = 400e-9
wl800 = 800e-9
wl266 = wl800/3;
# angular frequencies: (rad/s)
w400 = lib.twopi*lib.sol/wl400
w800 = lib.twopi*lib.sol/wl800
w266 = 3 * w800;

# file names for 400 + 800 -> 266 FWM
# first scan: no fused silica inserted into either beam
path0 = glob.glob("sample_data/20240313*2uJ800-26uJ400-grating" + grating + "-0cmFS")

# scans with fused silica inserted into 400:
path2 = glob.glob("sample_data/20240313*2uJ800-26uJ400-grating" + grating + "-2cmFSin400")

 # this is the case for wl2 == 266 nm
# file names for 400 + 800 -> 266 FWM
# first scan: no fused silica inserted into either beam
path3 = glob.glob("sample_data/20240313*10uJ266-10uJ400-grating" + grating + "-0cmFS")
# scans with fused silica inserted into 400:
path5 = glob.glob("sample_data/20240313*10uJ266-10uJ400-grating" + grating + "-2cmFSin400")
 
    

dirnames800 = [path0[0], path2[0]]
dirnames266 = [path3[0], path5[0]]


N = 128
ft = pypret.FourierTransform(N, dt=6e-15)

# calculate using xFROG method for comparison
ref_file = "sample_data/grating" + grating

xfrog_measurement_ref = create_frog_input.read_frog_scan(path0[0], ft, w266)
ref_pulse = create_frog_input.read_ref_pulse(ref_file, ft, w0=w800)
# res_pulse = two_pulse_retrieve.xfrog_retrieve("fwm2", xfrog_measurement_ref, ref_pulse, ft, test_w0=w400)


# calculate the two pulses using known dispersion

if (FS_beam==400):
    L_1 = np.array([0.0, 2*9.432e-3])
    L_2 = np.array([0.0, 0*9.432e-3])
else:
    L_1 = np.array([0.0, 0*9.432e-3])
    L_2 = np.array([0.0, 2*9.432e-3])



# known spectra of each pulse from spectrometer
fspec800 = path0[0] + "\\probe_only.txt"
fspec400 = path0[0] + "\\pump_only.txt"
fwls = path0[0] + "\\wavelengths.txt"

spec400_interp = create_frog_input.read_ref_spectra(ft, w400, fspec400, file_type="separate", wl_file=fwls)
spec800_interp = create_frog_input.read_ref_spectra(ft, w800, fspec800, file_type="separate", wl_file=fwls)

fspec266 = "sample_data\\20240313-FWM_ref_spectra\\grating%s.txt"%grating
spec266_interp = create_frog_input.read_ref_spectra(ft, w266, fspec266, file_type="combined")

xfrog_measurement800 = create_frog_input.create_combined_measurements(nscans, dirnames800, ref_insertions=L_1, test_insertions=L_2, ft=ft, process_w0=w266, pulse_w0s=[w400, w800], target_spec="UV-VIS")

# create inital guess pulses:
guess_pulse400 = pypret.Pulse(ft, wl400)
guess_pulse800 = pypret.Pulse(ft, wl800)

guess_pulse400.spectrum = spec400_interp
guess_pulse800.spectrum = spec800_interp


#%%
guess_spectra800 = np.concatenate((guess_pulse800.spectrum, guess_pulse400.spectrum))

pnps800 = pypret.PNPS(guess_pulse800, "2-pulse-xfrog", "fwm1", reference_pulse=guess_pulse400, nscans=nscans, test_insertion=L_2, ref_insertion=L_1)
ret800 = Retriever(pnps800, "2-pulse-copra", verbose=True, logging=True,
                    maxiter=100)
ret800.retrieve(xfrog_measurement800, guess_spectra800)

result800 = ret800.result()

res_pulse1 = pypret.Pulse(ft, wl400)
res_pulse2 = pypret.Pulse(ft, wl800)

res_pulse2.spectrum = result800.pulse_retrieved[0:N]
res_pulse1.spectrum = result800.pulse_retrieved[N:2*N]
#%%
w266 = lib.twopi*lib.sol/266e-9
# calculate 266 nm pulse
xfrog_measurement266 = create_frog_input.create_combined_measurements(nscans, dirnames266, ref_insertions=L_1, test_insertions=L_2, ft=ft, process_w0=(2*w400 - w266), pulse_w0s=[w400, w266], target_spec="UV-VIS")

guess_pulse266 = pypret.Pulse(ft, w266, unit='om')
random_gaussian(guess_pulse400, 80e-15)
random_gaussian(guess_pulse266, 80e-15)

guess_spectra266 = np.concatenate((guess_pulse266.spectrum, guess_pulse400.spectrum))

pnps266 = pypret.PNPS(guess_pulse266, "2-pulse-xfrog", "fwm1", reference_pulse=guess_pulse400, nscans=nscans, test_insertion=L_2, ref_insertion=L_1)
ret266 = Retriever(pnps266, "2-pulse-copra", verbose=True, logging=True,
                    maxiter=100)
ret266.retrieve(xfrog_measurement266, guess_spectra266)

result266 = ret266.result()

res_pulse3 = pypret.Pulse(ft, w400, unit='om')
res_pulse4 = pypret.Pulse(ft, w266, unit='om')

res_pulse4.spectrum = result266.pulse_retrieved[0:N]
res_pulse3.spectrum = result266.pulse_retrieved[N:2*N]

#%%

fig6, (ax9, ax10) = plt.subplots(2, 1, figsize=(6, 6))
plot_trace_comparison(xfrog_measurement800, result800.trace_retrieved, axes=(ax9, ax10))

# plt.suptitle("800 nm + 400 nm FWM")
ax9.set_ylim([255, 275])
ax10.set_ylim([255, 275])
plt.tight_layout()

#%%

fig5, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(8, 6))
plot_retrieved_field(res_pulse1, axes=(ax5, ax6))
plot_field_comparison(res_pulse2, ref_pulse, axes=(ax7, ax8))
for ax in (ax5, ax6, ax8):
    ax.legend().remove()
ax7.legend(labels=["_","SPIDER intensity", "_", "SPIDER phase"], fontsize=8, loc='upper left')
# ax5.set_title("400 nm pulse")
# ax6.set_title("400 nm spectrum")
# ax7.set_title("800 nm pulse")
# ax8.set_title("800 nm spectrum")
# plt.suptitle("Blind retrieval of 400 nm + 800 nm")
plt.tight_layout()

fig1, (ax11, ax12) = plt.subplots(2, 1, figsize=(6, 6))
plot_trace_comparison(xfrog_measurement266, result266.trace_retrieved, axes=(ax11, ax12))
# plt.suptitle("266 nm + 400 nm FWM")
plt.tight_layout()
ax11.set_ylim([720, 870])
ax12.set_ylim([720, 870])




fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
plot_retrieved_field(res_pulse4, axes=(ax1, ax2))
plot_field_comparison(res_pulse3, res_pulse1, axes=(ax3, ax4))


for ax in (ax1, ax2, ax4):
    ax.legend().remove()
ax3.legend(labels=["_","Ref. intensity", "_", "Ref. phase"], fontsize=8, loc='upper left')
# ax1.set_title("266 nm pulse")
# ax2.set_title("266 nm spectrum")
# ax3.set_title("400 nm pulse")
# ax4.set_title("400 nm spectrum")
for ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8):
    ax.set_title("")
# plt.suptitle("Blind retrieval of 400 nm + 266 nm")
plt.tight_layout()