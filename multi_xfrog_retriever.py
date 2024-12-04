# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:15:20 2024

@author: Noa


Modification of COPRA retriever to modify two pulses simultaneously
and to calculate separate values of the scale factor mu for each trace

"""

import numpy as np
from pypret import lib
from pypret.retrieval.step_retriever import StepRetriever
from delay_param import get_scan_num_array

    
class TwoPulseRetriever(StepRetriever):
    """ Modification of COPRA algorithm to include concurrent optimization of
    two pulses.
    Spectrum array is modified to have length 2*N, where 0:N is the spectrum of
    the first (test) pulse, and N:2N is the spectrum of the second (ref) pulse
    """
    method = "2-pulse-copra"

    def __init__(self, pnps, alpha=0.25, **kwargs):
        """ For a full documentation of the arguments see :class:`Retriever`.

        Parameters
        ----------
        alpha : float, optional
            Scales the step size in the global stage of COPRA. Higher values
            mean potentially faster convergence but less accuracy. Lower
            values provide higher accuracy for the cost of speed. Default is
            0.25.
        """
        super().__init__(pnps, alpha=alpha, **kwargs)

    def _retrieve_begin(self, measurement, initial_guess, weights):
        super()._retrieve_begin(measurement, initial_guess, weights)
        pnps = self.pnps
        rs = self._retrieval_state
        rs.mode = "local"  # COPRA starts with local mode
        # calculate the maximum gradient norm
        # self.trace_error() was called beforehand -> rs.Tmn and rs.Smk exist!
        Smk2 = self._project(self.Tmn_meas / rs.mu, rs.Smk)
        nablaZnm = pnps.gradient(Smk2, self.parameter)
        rs.current_max_gradient1 = np.max(np.sum(lib.abs2(nablaZnm[0:self.ft.N]), axis=1))
        rs.current_max_gradient2 = np.max(np.sum(lib.abs2(nablaZnm[self.ft.N:2*self.ft.N]), axis=1))

    def _retrieve_step(self, iteration, En):
        """ Perform a single COPRA step.

        Parameters
        ----------
        iteration : int
            The current iteration number - mainly for logging.
        En : 1d-array
            The current pulse spectrum.
        """
        # local rename
        ft = self.ft
        options = self.options
        pnps = self.pnps
        rs = self._retrieval_state
        Tmn_meas = self.Tmn_meas
        # current gradient -> last gradient
        rs.previous_max_gradient1 = rs.current_max_gradient1
        rs.current_max_gradient1 = 0.0
        # two gradients, to scale step differently for each pulse
        rs.previous_max_gradient2 = rs.current_max_gradient2
        rs.current_max_gradient2 = 0.0
        # switch iteration
        if rs.steps_since_improvement == 5:
            rs.mode = "global"
        # local iteration
        if rs.mode == "local":
            # running estimate for the trace
            Tmn = np.zeros((self.M, self.N))
            for m in np.random.permutation(np.arange(self.M)):
                p = self.parameter[m]
                Tmn[m, :] = pnps.calculate(En, p)
                Smk2 = self._project(Tmn_meas[m, :] / rs.mu[m, :], pnps.Smk)
                nablaZnm = pnps.gradient(Smk2, p)
                # calculate the step size
                Zm = lib.norm2(Smk2 - pnps.Smk)
                gradient_norm1 = lib.norm2(nablaZnm[0:ft.N])
                if gradient_norm1 > rs.current_max_gradient1:
                    rs.current_max_gradient1 = gradient_norm1
                gamma1 = Zm / max(rs.current_max_gradient1,
                                  rs.previous_max_gradient1)    
                    
                gradient_norm2 = lib.norm2(nablaZnm[ft.N:2*ft.N])
                if gradient_norm2 > rs.current_max_gradient2:
                    rs.current_max_gradient2 = gradient_norm2
                gamma2 = Zm / max(rs.current_max_gradient2,
                                  rs.previous_max_gradient2)

                
                En[0:ft.N] -= gamma1 * nablaZnm[0:ft.N]
                En[ft.N:2*ft.N] -= gamma2 * nablaZnm[ft.N:2*ft.N]
                
            

            # Tmn is only an approximation as En changed in the iteration!
            rs.approximate_error = True
            R = self._R(Tmn)  # updates rs.mu!!!
        # global iteration
        elif rs.mode == "global":
            Tmn = pnps.calculate(En, self.parameter)
            r = self._r(Tmn)
            R = self._Rr(r)  # updates rs.mu!!!
            rs.approximate_error = False
            # gradient descent w.r.t. Smk
            w2 = self._weights * self._weights
            gradrmk = (-4 * ft.dt / (ft.dw * lib.twopi) *
                       ft.backward(rs.mu * ft.forward(pnps.Smk) *
                                   (Tmn_meas - rs.mu * Tmn) * w2))
            etar = options.alpha * r / lib.norm2(gradrmk)
            Smk2 = pnps.Smk - etar * gradrmk
            # gradient descent w.r.t. En
            nablaZn = pnps.gradient(Smk2, self.parameter).sum(axis=0)
            # calculate the step size
            Z = lib.norm2(Smk2 - pnps.Smk)
            etaz = options.alpha * Z / lib.norm2(nablaZn[0:ft.N])
            # update the spectrum
            En[0:ft.N] -= etaz * nablaZn[0:ft.N]
        return R, En
    
    # override the calculation of rs.mu to return a vector which has different 
    # values for different m (i.e. different traces)
    def _error_vector(self, Tmn, store=True):
        """ Calculates the residual vector from measured to simulated
        intensity.
        """
        # rename
        rs = self._retrieval_state
        Tmn_meas = self.Tmn_meas
        scan_nums = get_scan_num_array(self.parameter)
        mu = np.ndarray((scan_nums.size, 1))
        # scaling factor
        w2 = self._weights * self._weights
        for i in range(self.pnps.nscans):
            inds = (scan_nums == i)
            mu[inds] = np.sum(Tmn_meas[inds, :] * Tmn[inds, :] * w2[inds]) / np.sum(Tmn[inds, :] * Tmn[inds,:] * w2[inds])
        # store intermediate results in current retrieval state
        if store:
            rs.mu = mu
            rs.Tmn = Tmn
            rs.Smk = self.pnps.Smk
        return np.ravel((Tmn_meas - mu * Tmn) * self._weights)