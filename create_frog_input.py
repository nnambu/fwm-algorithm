"""

Some custom functions for reading expreimental data and converting to MeshData
format used by pypret

These are specific to the format in which data was gathered and saved - for
different experimental setups new functions are needed

"""
import pypret
from pypret.mesh_data import (MeshData, lib)
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
import warnings
from scipy.optimize import curve_fit
from dispersion_delay_test import FS_ngroup  # import this in order to calculate group velocity
from pypret import material
import delay_param


def read_frog_scan(dirname, ft, w0, target_spec="UV-VIS", return_delay=False):
    """
    dirname: path to folder containing data files for a given 
    ft: pypret.FourierTransform object containing time and frequency grids
    w0: center frequency of target process
    target_spec: which spectrometer to look for if the data was taken using two connected spectrometers
        options: "UV-VIS", "VIS-NIR"
    """

    # dirname = "D:\\FWM_XFROG\\20230322-145723-20uJ400-2uJ800-grating5300-2cmFSin800"
    
    scan_data = np.loadtxt(dirname + "\\spectra.txt")
    wavelengths = np.loadtxt(dirname + "\\wavelengths.txt")
    motorpos = np.loadtxt(dirname + "\\motor_positions.txt")
    
    if target_spec=="UV-VIS":
        if not ((np.abs(np.max(wavelengths)-889.704059303079474)<1) and (np.abs(np.min(wavelengths)-180.402739999999994) < 1)):
            wavelengths = np.loadtxt(dirname + "\\wavelengths2.txt")
            scan_data = np.loadtxt(dirname + "\\spectra2.txt")
    elif target_spec=="VIS-NIR":
        if not ((np.abs(np.max(wavelengths)-1004.212984905690630)<1) and (np.abs(np.min(wavelengths)-320.422520000000020) < 1)):
            wavelengths = np.loadtxt(dirname + "\\wavelengths2.txt")
            scan_data = np.loadtxt(dirname + "\\spectra2.txt")
            wavelengths = wavelengths[0:2048]
    
    # use manual calibration of spectrometer
# =============================================================================
#     if wavelengths.size == 2048:
#         wi = np.arange(2048)
#         wavelengths = -1.145085e-9*(wi**3) - 2.0922449e-5*(wi**2) + 0.38232277*wi + 319.354997
# =============================================================================
        


    # if the process is fwm2, then delay of reference pulse (266 or 800) increases
    # as motorposition decreases, so we need to flip the scan - this can be done in the algorithm for each iteration
    # scan_data = np.flip(scan_data, 0)
    
    # use the first scan as a background (assuming there is no FWM signal at this delay)
    background = scan_data[0, :]
    scan_data = scan_data - background
    
    # crop out longer/shorter wavelengths before removing noise
    signal_indices = (wavelengths > 2*np.pi*2.9979245e17/(ft.w[-1]+w0)) & (wavelengths < 2*np.pi*2.9979245e17/(ft.w[0]+w0))
    scan_data = scan_data[:, signal_indices]
    wavelengths = wavelengths[signal_indices]
    

    
    # convert motor position in mm to delay in seconds, and center around the maximum signal
    delays = motorpos * 2 / 2.99792458e11    # s
    lineout = np.sum(scan_data[:, :], axis=1)

    t_cog = np.trapz(lineout * delays)/np.trapz(lineout)
    delays = delays - t_cog
    
    # remove the DC backgroud that increases linearly with delay
    # For now: just use the first and last points to do a fit
    DC_noise = (delays - delays[0]) * (lineout[-1]-lineout[0]) / (delays[-1]-delays[0])

    # plt.plot(delays, DC_noise)
    lineout = lineout - DC_noise    
    
    DC_noise = DC_noise / scan_data.shape[1]
    
    scan_data = scan_data - DC_noise[:, np.newaxis]
    
    # crop the box to fit the FWM signal
    zero_crossings = np.where(np.diff(np.signbit(lineout - np.max(lineout)*0.1)))[0];
    cutoffs = delays[zero_crossings] * 2
    scan_data = scan_data[(delays >= cutoffs[0]) & (delays <= cutoffs[1]), :]

    delays = delays[(delays >= cutoffs[0]) & (delays <= cutoffs[1])]
    


    
    # remove noise below a certain threshold
    scan_data = scan_data / np.max(scan_data)
    scan_data[scan_data < 0.01] = 0
    
    
    # plot spectrum vs frequency instead of wavelength
    omegas = 2*np.pi*2.998e17 / wavelengths     # rad/s
    scan_data = scan_data * wavelengths * wavelengths
    scan_data = np.flip(scan_data, 1)
    omegas = np.flip(omegas)
    
    # normalize to a max value of 1
    scan_data = scan_data / np.max(scan_data)

    
    # Interpolate the data onto a square grid centered around the FWM signal
    w_pts = w0 + ft.w
    t_pts = np.linspace(delays[0], delays[-1], ft.N)
    
    interp_mesh = np.array(np.meshgrid(t_pts, w_pts))
    interp_points = np.rollaxis(interp_mesh, 0, 3)
    
    scan_interp = interpn((delays, omegas), scan_data, interp_points, bounds_error=False, fill_value=0) 

    
    
    mdata = MeshData(scan_interp.T, t_pts, w_pts)
    
    if return_delay:
        return mdata, t_cog
    else:
        return mdata

# read pulse from SPIDER input
def read_ref_pulse(file_path, ft, w0=None):
    # file_path = "D:\\20230321_SPIDER_800nm\\with_lens\\grating4800_noFS"
    time_data = np.loadtxt(file_path + "_time.dat", skiprows=1)
    freq_data = np.loadtxt(file_path + "_freq.dat", skiprows=1)
    COG800 = np.loadtxt(file_path + "_values.dat", skiprows=1, max_rows=1, usecols=(-1,))   # center wavelength in nm
    if w0 == None:
        pulse = pypret.pulse.Pulse(ft, COG800*1e-9)
    else:
        pulse = pypret.pulse.Pulse(ft, w0, unit='om')
    
    t = time_data[:, 0]*1e-15
    I = time_data[:, 2]
    phi = time_data[:, 3]
    E = np.sqrt(I)   

    
    # send a warning if some of the field is being cut off
    t_cut = (t < ft.t[0]) | (t > ft.t[-1])
    if (np.sum(t_cut) > 0) and (np.max(I[t_cut]) > 1e-2):
        warnings.warn("Part of reference pulse is cut off!")
    
    # interpolate onto the given grid
    E_interp = np.interp(ft.t, t, E)
    phi_interp = np.interp(ft.t, t, phi)

    
        
    
    pulse.field = E_interp * np.exp(1.0j*phi_interp)


    return pulse


def create_combined_measurements(nscans, dirnames, ref_insertions, test_insertions, ft, process_w0, pulse_w0s, target_spec="UV-VIS"):
    """
    pulse 1 is 400 nm reference, pulse 2 is 800/266
    dirname: path to folder containing data files for a given 
    ft: pypret.FourierTransform object containing time and frequency grids
    w0: center frequency of target process
    target_spec: which spectrometer to look for if the data was taken using two connected spectrometers
        options: "UV-VIS", "VIS-NIR"
    process_w0: center frequency of the process
    pulse_w0s: center frequencies of the two pulses (used to calculated group delay)
    
    Experimental scans are assuming FWM1 process (400 nm reference (E(t - tau)^2) and 800/266 nm test pulse (E*(t)))
    """
    
    ref_insertions = np.array(ref_insertions)
    test_insertions = np.array(test_insertions)
    N = ft.N
    combined_data = np.ndarray((nscans*N,N))
    combined_delays = np.ndarray((nscans*N,))
    scan_nums = np.ndarray((nscans*N,))
    center_delays = np.zeros((nscans,))
    

    
    for i in range(nscans):
        scan_nums[(N*i):(N*i + N)] = i
        current_scan, current_delay = read_frog_scan(dirnames[i], ft, process_w0, target_spec=target_spec, return_delay=True)
        center_delays[i] = current_delay
        combined_data[(N*i):(N*i + N), :] = current_scan.data
        combined_delays[(N*i):(N*i + N)] = current_delay + current_scan.axes[0]
        # combined_delays[(N*i):(N*i + N)] = -added_delays[i] + current_scan.axes[0]
    
    combined_delays = combined_delays - center_delays[0]

    center_delays = center_delays - center_delays[0]

    combined_parameter = delay_param.create_param_array(combined_delays, scan_nums)
    mdata = MeshData(combined_data, combined_parameter, current_scan.axes[1])
    return mdata
    
    
        
def read_ref_spectra(ft, w_center, spec_file, file_type, wl_file=""):
    """
    ft: the fourier transform object provides frequency grid to interpolate onto
    w_center: the center frequency (rad/s) of the grid to interpolate onto
    spec_file: path to the file containing spectrum data (either spectra only or spectra + wavelengths)
    file_type: 2 choices
        "combined": both wavelengths and spectrum is in the same file (spec_file) - data from Oceanview (contains header line)
        "separate": wavelengths are in separate file (wl_files) - data from Labview
    """
    if file_type=="combined":
        spec_data = np.loadtxt(spec_file, skiprows=1)
        measured_wls = spec_data[:, 0]
        spec_meas = spec_data[:, 1]
    else:
        spec_meas = np.loadtxt(spec_file, skiprows=0)
        measured_wls = np.loadtxt(wl_file)
        measured_wls = measured_wls[0:spec_meas.size]
    # remove the background from spectrometer measurements
    # to remove background: assume values within 10% and 90% of data are all background
    # then polyfit those values
    # repeat 3 times for better fit
    for i in range(3):
        per = np.percentile(spec_meas, (10, 90))
        bg_indices = (spec_meas > per[0]) & (spec_meas < per[1])
        p1 = np.polyfit(measured_wls[bg_indices], spec_meas[bg_indices], 2)
        bglevel = np.polyval(p1, measured_wls)
        spec_meas = spec_meas - bglevel
    
    # interpolate onto the provided grid
    spec_interp = np.interp(ft.w + w_center, 2*np.pi*2.998e17/np.flip(measured_wls), np.flip(spec_meas * measured_wls**2))
    spec_interp = spec_interp / np.max(spec_interp)     
    return spec_interp