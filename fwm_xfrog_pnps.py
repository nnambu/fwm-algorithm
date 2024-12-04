"""

Custom PNPS class to include calculation and gradients of two-pulse XFROG
process

FWM XFROG trace and gradients are calculated for difference-frequency FWM process

"""

import pypret
import numpy as np
from pypret import lib
from pypret import io
from pypret import material
from pypret.mesh_data import MeshData
from pypret.frequencies import convert
from pypret.pulse import Pulse
from pypret.pnps import NoncollinearPNPS

# for multiXFROG, use custom object as parameter which contains scan_no and delay
import delay_param

    

class TwoPulseXFROG(NoncollinearPNPS):
    """
    PNPS class that takes two test pulses and calculates a spectrum
    and gradients
    
    Input spectrum is stitched together array of length 2N
    """
    
    # fwm1: |E_r|^2 x E_t*
    # fwm2: |E_t|^2 x E_r*
    _supported_processes = ["fwm2", "fwm1"]
    method = "2-pulse-xfrog"
    parameter_name = "delay"
    parameter_unit = "s"

    def __init__(self, pulse, process, reference_pulse: Pulse, nscans=2, test_insertion=[0, 0.01], ref_insertion=[0, 0]):
        """ Creates the instance.

        Parameters
        
        !!
        when calling calculate() and gradient(), 
        
        !!
        
        pulse : Pulse instance
            The pulse object that defines the simulation grid.
        process : str
            The nonlinear process used in the PNPS method.
        reference_pulse: the XFROG reference pulse, must be passed in as a keyword argument
        nscans: number of scans with different amounts of fused silica inserted. The delay (parameter)
            will be repeated nscans times (m goes from 0 to nscans(ndelays) - 1)
        test_insertion: list of length nscans specifying the amount of glass (in m) inserted into
            the test pulse for each scan
        ref_insertion: list of length nscans specifying the amount of glass (in m) inserted into
            the reference pulse for each scan
        """

        # base __init__ function creates an attribute called reference_pulse and assigns it the value of reference_pulse
        # initial call to PNPS must have an argument reference_pulse = [Pulse object]
        # scale_factors: scale the total energy in each scan so that they are all equal (to match normalized measurements)
        super().__init__(pulse, process, reference_pulse=reference_pulse, nscans=nscans, test_insertion=test_insertion,
                         ref_insertion=ref_insertion, scale_factors=np.ones((nscans,)))

    def _get_tmp(self, parameter):
        if parameter in self._tmp:
            return self._tmp[parameter]
        
        delay = np.exp(1.0j * parameter.delay * self.ft.w)
        N = self.ft.N
        scan_no = parameter.scan_no
        Ak = np.zeros(N, dtype=np.complex128)
        Ek = np.zeros(N, dtype=np.complex128)
        Sk = np.zeros(N, dtype=np.complex128)
        Tn = np.zeros(N, dtype=np.float64)
        # calculate the extra phase added by insterting known amounts of glass
        
        ref_dispersion = np.exp(1.0j*(material.FS.k(self.ft.w + self.reference_pulse.w0, unit='om')-(self.ft.w + self.reference_pulse.w0)/lib.sol)*self.ref_insertion[scan_no])

        test_dispersion = np.exp(1.0j*(material.FS.k(self.ft.w + self.w0, unit='om')-(self.ft.w + self.w0)/lib.sol)*self.test_insertion[scan_no]) 
        

        
        # ref_dispersion = np.exp(1.0j * 0.4671*self.ref_insertion[scan_no]/lib.sol * self.ft.w)
        self._tmp[parameter] = delay, Ak, Ek, Sk, Tn, scan_no, ref_dispersion, test_dispersion
        return delay, Ak, Ek, Sk, Tn, scan_no, ref_dispersion, test_dispersion

    def _calculate(self, spectrum, parameter):
        """ Calculates the nonlinear process spectrum for a single parameter.

        Follows the notation from our paper.
        """
        ft = self.ft
        delay, Ak, Ek, Sk, Tn, scan_no, ref_dispersion, test_dispersion = self._get_tmp(parameter)
        
        N = self.ft.N
        
        ft.backward(ref_dispersion * delay * spectrum[N:2*N], out=Ak)
        ft.backward(test_dispersion * spectrum[0:N], out=Ek)
        if self.process == "fwm1":
            Sk[:] = Ak * Ak * Ek.conj()
        elif self.process == "fwm2":
            Sk[:] = Ek * Ek * Ak.conj()
        Tn[:] = self.measure(Sk)
        Tn[:] = self.measure(Sk)
        return Tn, Sk


    def _gradient(self, Sk2, parameter):
        """ Returns the gradient of Z based on the previous call to _spectrum.
            Sk2 is the measured trace
        """
        ft = self.ft
        N = ft.N
        # retrieve the intermediate results
        delay, Ak, Ek, Sk, Tn, scan_no, ref_dispersion, test_dispersion = self._tmp[parameter]
        # difference between original and updated PNPS signal
        dSk = Sk2 - Sk
        # calculate the gradients for both pulses and combine them
        gradnZ = np.zeros((2*N, ), dtype=complex)
        if self.process == "fwm1":
            gradnZ[0:N] = ft.forward(dSk.conj() * Ak * Ak) * test_dispersion.conj()
            gradnZ[N:2*N] = 2*ft.forward(dSk * Ek * Ak.conj()) * ref_dispersion.conj() * delay.conj()
        if self.process == "fwm2":
            gradnZ[0:N] = 2*ft.forward(dSk * Ak * Ek.conj()) * test_dispersion.conj()
            gradnZ[N:2*N] = ft.forward(dSk.conj() * Ek * Ek) * ref_dispersion.conj() * delay.conj()

        # common scale for all gradients (note the minus)
        gradnZ *= -2.0 * lib.twopi * ft.dw / ft.dt
        return gradnZ
    
    
    def gradient(self, Smk2, parameter):
        """ Calculates the gradient ∇_n Z_m.
        """
        parameter = np.atleast_1d(parameter)
        Smk2 = np.atleast_2d(Smk2)
        gradnZm = np.zeros((parameter.shape[0], self.ft.N*2),
                           dtype=np.complex128)
        for m, p in enumerate(parameter.flat):
            gradnZm[m, :] = self._gradient(Smk2[m, :], p)
        # if a scalar parameter is passed, squeeze out one dimension
        return gradnZm.squeeze()
    
    
    def calculate(self, spectrum, parameter):
        """ Calculates the PNPS signal S_mk and the PNPS trace T_mn.

        Parameters
        ----------
        spectrum : 1d-array
            The pulse spectrum for which the PNPS trace is calculated.
        parameter : scalar or 1d-array
            The PNPS parameter (array) for which the PNPS trace is calculated.

        Returns
        -------
        1d- or 2d-array
            Returns the calculated PNPS trace over the frequency
            ``self.process_w``. If parameter was a scalar a 1d-array is
            returned. If it was a 1d-array a 2d-array is returned where the
            parameter runs along the first axis and the frequency along the
            second.
        """
        parameter = np.atleast_1d(parameter)
        Tmn = np.zeros((parameter.size, self.ft.N))
        Smk = np.zeros((parameter.size, self.ft.N), dtype=np.complex128)
        for m, p in enumerate(parameter):
            Tmn[m, :], Smk[m, :] = self._calculate(spectrum, p)
        # if a scalar parameter was used, squeeze out one dimension
        Tmn = Tmn.squeeze()
        Smk = Smk.squeeze()
        # store for later use (in self.trace)
        self.Tmn = Tmn
        self.Smk = Smk
        self.parameter = parameter
        self.spectrum = spectrum
        return Tmn