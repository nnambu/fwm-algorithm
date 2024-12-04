# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:45:00 2024

@author: Noa

Some functions to plot the results of FWM XFROG measurements

"""

import numpy as np
import matplotlib.pyplot as plt
import pypret
from pypret import (Retriever, FourierTransform, Pulse, random_pulse, random_gaussian, PNPS, MeshDataPlot, MeshData, lib)
from delay_param import get_delays_array, get_scan_num_array


def plot_field_comparison(original_pulse, retrieved_pulse, axes=None):
    # optionally you can pass the axes on which to plot - for example onto subplots of an existing figure
    if axes==None:
        f, (ax1, ax2) = plt.subplots(1, 2)
    else:
        ax1 = axes[0]
        ax2 = axes[1]
        f = ax1.get_figure()    
    ft = original_pulse.ft
    
    fsize = 14
    
    # shift pulses so that they are centered at zero in time
    tmax = np.argmax(retrieved_pulse.intensity)
    tmax = retrieved_pulse.ft.t[tmax]
    retrieved_pulse.spectrum = retrieved_pulse.spectrum*np.exp(-1j*tmax*retrieved_pulse.ft.w)
    tmax2 = np.argmax(original_pulse.intensity)
    tmax2 = original_pulse.ft.t[tmax2]
    original_pulse.spectrum = original_pulse.spectrum*np.exp(-1j*tmax2*original_pulse.ft.w)
    
    ax1.plot(original_pulse.t*1e15, original_pulse.intensity/np.max(original_pulse.intensity), 'k', label="Intensity (original)")
    ax1.plot(retrieved_pulse.t*1e15, retrieved_pulse.intensity/np.max(retrieved_pulse.intensity), '--', label="Intensity (retrieved)", linewidth=2.0)
    ax1.set_xlabel("t (fs)", fontsize=fsize)
    
    # ignore phase when intensity is below 5% of max value
    threshold = 0.05
    t_original = original_pulse.intensity/np.max(original_pulse.intensity) > threshold
    t_retrieved = retrieved_pulse.intensity/np.max(retrieved_pulse.intensity) > threshold

    ax1p = ax1.twinx()
    phase_t = retrieved_pulse.phase
    # phase_t[res_pulse.intensity < 0.01] = 0
    ax1p.plot(original_pulse.t[t_original]*1e15, original_pulse.phase[t_original] - original_pulse.phase[int(ft.N/2)], 'r', label="Phase (original)")
    ax1p.plot(retrieved_pulse.t[t_retrieved]*1e15, phase_t[t_retrieved] - phase_t[int(ft.N/2)], 'g:', label="Phase (retrieved)",linewidth=3.0)
    ax1.set_title("Time domain, FWHM = %d fs"%(retrieved_pulse.fwhm(dt=1e-15)*1e15))
    ax1.set_ylabel("Intensity (a.u.)", fontsize=fsize)
    ax1p.set_ylabel("Phase (rad)", fontsize=fsize)
    ax1p.set_ylim(-5, 5)

    # Create dummy lines for the legend
    ax1.plot([], [], 'r', label="Phase (original)")
    ax1.plot([], [], 'g:', label="Phase (retrieved)")
    ax1.legend(loc="upper left")


    # ax2.plot((original_pulse.w + original_pulse.w0)*1e-15, original_pulse.spectral_intensity/np.max(original_pulse.spectral_intensity), 'k', label="Intensity (original)")
    # ax2.plot((original_pulse.w + original_pulse.w0)*1e-15, retrieved_pulse.spectral_intensity/np.max(retrieved_pulse.spectral_intensity), '--', label="Intensity (retrieved)", linewidth=2.0)
    # ax2.set_xlabel("$\omega$ (rad/s x $10^{15}$)", fontsize=fsize)
    orig_spec = original_pulse.spectral_intensity * (original_pulse.w + original_pulse.w0) ** 2;
    orig_spec = orig_spec/np.max(orig_spec)
    ret_spec = retrieved_pulse.spectral_intensity * (original_pulse.w + original_pulse.w0) ** 2;
    ret_spec = ret_spec/np.max(ret_spec)
    ax2.plot(2*np.pi*2.998e8/(original_pulse.w + original_pulse.w0)*1e9, orig_spec, 'k', label="Intensity (original)")
    ax2.plot(2*np.pi*2.998e8/(original_pulse.w + original_pulse.w0)*1e9, ret_spec, '--', label="Intensity (retrieved)", linewidth=2.0)
    ax2.set_xlabel("Wavelength (nm)", fontsize=fsize)
    
    # ignore phase when intensity is below 5% of max value
    threshold = 0.05
    w_original = original_pulse.spectral_intensity/np.max(original_pulse.spectral_intensity) > threshold
    w_retrieved = retrieved_pulse.spectral_intensity/np.max(retrieved_pulse.spectral_intensity) > threshold
    
    ax2p = ax2.twinx()
    phase_w = retrieved_pulse.spectral_phase - retrieved_pulse.spectral_phase[int(ft.N/2)]

    # ax2p.plot((original_pulse.w[w_original] + original_pulse.w0)*1e-15, original_pulse.spectral_phase[w_original] - original_pulse.spectral_phase[int(ft.N/2)], 'r', label="Phase (original)")
    # ax2p.plot((retrieved_pulse.w[w_retrieved] + retrieved_pulse.w0)*1e-15, phase_w[w_retrieved], 'g:', label="Phase (retrieved)", linewidth=3.0)   
    ax2p.plot(2*np.pi*2.998e8/(original_pulse.w[w_original] + original_pulse.w0)*1e9, original_pulse.spectral_phase[w_original] - original_pulse.spectral_phase[int(ft.N/2)], 'r', label="Phase (original)")
    ax2p.plot(2*np.pi*2.998e8/(retrieved_pulse.w[w_retrieved] + retrieved_pulse.w0)*1e9, phase_w[w_retrieved], 'g:', label="Phase (retrieved)", linewidth=3.0)
    # ax1.set_title("Frequency domain")
    
    ax2.set_ylabel("Intensity (a.u.)", fontsize=fsize)
    ax2p.set_ylabel("Spectral phase (rad)", fontsize=fsize)
    ax2p.set_ylim(-5, 5)
    
    # Create dummy lines for the legend
    ax2.plot([], [], 'r', label="Spectral phase (original)")
    ax2.plot([], [], 'g:', label="Spectral phase (retrieved)")
    ax2.legend(loc="upper left")
    
    ax2p.yaxis.label.set_color('r')
    ax2p.spines['right'].set_color('r')
    ax2p.tick_params(axis='y', colors='r')
    
    ax1p.yaxis.label.set_color('r')
    ax1p.spines['right'].set_color('r')
    ax1p.tick_params(axis='y', colors='r')


    
    return f

def plot_retrieved_field(retrieved_pulse, axes=None):
    # optionally you can pass the axes on which to plot - for example onto subplots of an existing figure
    if axes==None:
        f, (ax1, ax2) = plt.subplots(1, 2)
    else:
        ax1 = axes[0]
        ax2 = axes[1]
        f = ax1.get_figure()
    
    ft = retrieved_pulse.ft
    
    fsize = 14
    
    # shift pulse so that it is centered at zero in time
    tmax = np.argmax(retrieved_pulse.intensity)
    tmax = retrieved_pulse.ft.t[tmax]
    retrieved_pulse.spectrum = retrieved_pulse.spectrum*np.exp(-1j*tmax*retrieved_pulse.ft.w)
    
    # calculate GDD
    threshold = 0.05
    pha_indices = retrieved_pulse.spectral_intensity/np.max(retrieved_pulse.spectral_intensity) > threshold
    fit = np.polyfit(retrieved_pulse.w[pha_indices], retrieved_pulse.spectral_phase[pha_indices], 2, 
                      w=retrieved_pulse.spectral_intensity[pha_indices]**(0.5))
    GDD = 2*fit[0]
    phase_fit =  np.polyval(fit, retrieved_pulse.w)


    t_indices = retrieved_pulse.intensity/np.max(retrieved_pulse.intensity) > threshold
    
    l1 = ax1.plot(retrieved_pulse.t*1e15, retrieved_pulse.intensity/np.max(retrieved_pulse.intensity), 'k', label="Intensity (retrieved)")
    ax1.set_xlabel("t (fs)", fontsize=fsize)
    ax1p = ax1.twinx()
    phase_t = retrieved_pulse.phase
    # phase_t[res_pulse.intensity < 0.01] = 0
    l2 = ax1p.plot(retrieved_pulse.t[t_indices]*1e15, phase_t[t_indices] - phase_t[int(ft.N/2)], color='r', label="Phase (retrieved)")
    ax1.set_title("Time domain, FWHM = %d fs"%(retrieved_pulse.fwhm(dt=1e-15)*1e15))
    # plt.legend([l1+l2], ["Intensity", "Phase"])
    ax1.set_ylabel("Intensity (a.u.)", fontsize=fsize)
    ax1p.set_ylabel("Phase (rad)", fontsize=fsize)
    ax1p.set_ylim(-5, 5)
    
    # Create a dummy line for the legend
    ax1.plot([], [], 'r', label="Phase (retrieved)")
    ax1.legend()



    # ax2.plot((retrieved_pulse.w + retrieved_pulse.w0)*1e-15, retrieved_pulse.spectral_intensity/np.max(retrieved_pulse.spectral_intensity), 'k', label="Intensity (retrieved)")
    # ax2.set_xlabel("$\omega$ (rad/s x $10^{15}$)", fontsize=fsize)
    ret_spec = retrieved_pulse.spectral_intensity * (retrieved_pulse.w + retrieved_pulse.w0)**2;
    ret_spec = ret_spec/np.max(ret_spec)
    ax2.plot(2*np.pi * 2.998e8/(retrieved_pulse.w + retrieved_pulse.w0)*1e9, ret_spec, 'k', label="Intensity (retrieved)")
    ax2.set_xlabel("Wavelength (nm)", fontsize=fsize)
    ax2p = ax2.twinx()
    phase_w = retrieved_pulse.spectral_phase - retrieved_pulse.spectral_phase[int(ft.N/2)]
    # ax2p.plot((retrieved_pulse.w[pha_indices] + retrieved_pulse.w0)*1e-15, phase_w[pha_indices], color='r', label="Phase (retrieved)")
    ax2p.plot(2*np.pi*2.998e8/(retrieved_pulse.w[pha_indices] + retrieved_pulse.w0)*1e9, phase_w[pha_indices], color='r', label="Phase (retrieved)")
    ax2.set_title("Frequency domain, GDD = %d $fs^2$"%(GDD*1e30))
    # ax2p.plot(retrieved_pulse.w + retrieved_pulse.w0, phase_fit - phase_fit[int(ft.N/2)], '.')
    
    ax2.set_ylabel("Intensity (a.u.)", fontsize=fsize)
    ax2p.set_ylabel("Spectral phase (rad)", fontsize=fsize)
    
    # color the right yaxis red
    ax2p.yaxis.label.set_color('r')
    ax2p.spines['right'].set_color('r')
    ax2p.tick_params(axis='y', colors='r')
    
    ax1p.yaxis.label.set_color('r')
    ax1p.spines['right'].set_color('r')
    ax1p.tick_params(axis='y', colors='r')


    ax2p.set_ylim(-5, 5)
    
    # create a dummy line for the legend
    ax2.plot([], [], color='r', label="Phase (retrieved)")
    ax2.legend()

    
    return f

def plot_trace_comparison(measured_trace, retrieved_object, pnps=None, axes=None):
    # retrieved_object is either a pulse, a MeshData trace, or a np array trace
    # optionally you can pass the axes on which to plot - for example onto subplots of an existing figure
    if axes is None:
        f, (ax1, ax2) = plt.subplots(1, 2)
    else:
        ax1 = axes[0]
        ax2 = axes[1]
        f = ax1.get_figure()
        
    
    parameter = measured_trace.axes[0]
    frequencies = measured_trace.axes[1]
    
    if pnps is not None:
        pnps.calculate(retrieved_object.spectrum, parameter)
        retrieved_trace = pnps.trace.data
    elif isinstance(retrieved_object, pypret.MeshData):
        retrieved_trace = retrieved_object.data
    elif isinstance(retrieved_object, np.ndarray):
        retrieved_trace = retrieved_object
    else:
        print("Unknown data type passed for retrieved trace")
    
    
    if parameter.dtype == 'float':
        delays = parameter * 1e15
    else:
        delays = get_delays_array(parameter) * 1e15
        nscans = parameter[-1].scan_no + 1
        scan_no = get_scan_num_array(parameter)
        
        # normalize each trace independently
        for i in range(nscans):
            measured_trace.data[scan_no==i, :] /= np.max(measured_trace.data[scan_no==i, :])
            retrieved_trace[scan_no==i, :] /= np.max(retrieved_trace[scan_no==i, :])


    
    # plot vertical lines to divide the different scans
    dividers = np.linspace(1/nscans, (nscans-1)/nscans, nscans-1)
    
    xticks = np.linspace(1/(6*nscans),(6*nscans-1)/(6*nscans),nscans*3)
    xtickvals = np.interp(xticks, np.linspace(0, (delays.size-1)/delays.size, delays.size), delays)
    if (np.abs(xtickvals[1]) < 20):
        xtickvals[0:3] -= xtickvals[1]
    print(xticks)
    xticklabels = []
    for val in xtickvals:
        xticklabels.append("%.0f"%val)
    
    fsize = 14
    
    # interpolate traces onto a wavelength scale
    wavelengths = np.linspace(2*np.pi*2.98e8/frequencies[-1], 2*np.pi*2.98e8/frequencies[0], frequencies.size)
    measured_interp = np.ndarray(measured_trace.data.T.shape)
    retrieved_interp = np.ndarray(retrieved_trace.T.shape)
    for i in range(0, delays.size):
        measured_interp[:, i] = np.interp(wavelengths, 2*np.pi*2.98e8/np.flip(frequencies), np.flip(measured_trace.data.T[:, i]))
        measured_interp[:, i] = measured_interp[:, i] / wavelengths**2
        retrieved_interp[:, i] = np.interp(wavelengths, 2*np.pi*2.98e8/np.flip(frequencies), np.flip(retrieved_trace.T[:, i]))
        retrieved_interp[:, i] = retrieved_interp[:, i] / wavelengths**2


        
    plt.axes(ax1)
    ax1.imshow(measured_interp, origin='lower', extent=(0, 1, 
            wavelengths[0]*1e9, wavelengths[-1] * 1e9), aspect='auto')
    ax1.set_xlabel("Delay (fs)", fontsize=fsize)
    ax1.set_ylabel("Wavelength (nm)", fontsize=fsize)
    # plt.title("Measured")
    if nscans > 1:
        plt.axvline(dividers, color='w')
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels)
    
    plt.axes(ax2)
    ax2.imshow(retrieved_interp, origin='lower', extent=(0, 1, 
            wavelengths[0]*1e9, wavelengths[-1]*1e9), aspect='auto')
    ax2.set_xlabel("Delay (fs)", fontsize=fsize)
    ax2.set_ylabel("Wavelength (nm)", fontsize=fsize)
    # plt.title("Retrieved")
    if nscans > 1:
        plt.axvline(dividers, color='w')
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels)



    
    return f
    


def plot_single_trace(measured_trace, frequencies=None, parameter=None, axes=None):
    # retrieved_object is either a MeshData trace, or a np array trace
    # if it is an array, then axes (frequency and delay parameter) also needed
    # optionally you can pass the axes on which to plot - for example onto subplots of an existing figure
    if axes is None:
        f = plt.figure()
        ax1 = f.add_subplot(111)
    else:
        ax1 = axes
        f = ax1.get_figure()
        
    if frequencies is not None:
        frequencies = frequencies*1e-15

        
    if isinstance(measured_trace, pypret.MeshData):
        frequencies = measured_trace.axes[1] * 1e-15
        parameter = measured_trace.axes[0]
        measured_trace = measured_trace.data

    
    
    scan_nos = np.zeros(parameter.shape)
    if parameter.dtype == 'float':
        delays = parameter * 1e15
    else:
        delays = get_delays_array(parameter) * 1e15
        nscans = parameter[-1].scan_no + 1
        scan_nos = get_scan_num_array(parameter)
    
    # normalize each trace independently
    for i in range(nscans):
        measured_trace[scan_nos==i, :] /= np.max(measured_trace[scan_nos==i, :])
    
    # plot vertical lines to divide the different scans
    dividers = np.linspace(1/nscans, (nscans-1)/nscans, nscans-1)
    
    xticks = np.linspace(1/(6*nscans),(6*nscans-1)/(6*nscans),nscans*3)
    xtickvals = np.interp(xticks, np.linspace(0, (delays.size-1)/delays.size, delays.size), delays)
    xticklabels = []
    for val in xtickvals:
        xticklabels.append("%.0f"%val)
    
    
    plt.axes(ax1)
    ax1.imshow(measured_trace.T, origin='lower', extent=(0, 1, 
            frequencies[0], frequencies[-1]), aspect='auto')
    ax1.set_xlabel("Delay (fs)")
    ax1.set_ylabel("$\omega$ (rad/s x$10^{15}$)")
    plt.title("FROG trace")
    if nscans > 1:
        plt.axvline(dividers, color='w')
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels)
    
    return f
    