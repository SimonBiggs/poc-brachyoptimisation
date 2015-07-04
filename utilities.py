# Copyright (C) 2015 Simon Biggs
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public
# License along with this program. If not, see
# http://www.gnu.org/licenses/.


import numpy as np
from scipy.optimize import basinhopping


class BasinhoppingWrapper(object):

    def __init__(self, n=5, 
                 optimiser_confidence=0.00001, 
                 basinhopping_confidence=0.00001, 
                 debug=None, bounds=None, **kwargs):
        self.to_minimise = kwargs['to_minimise']
        self.n = n
        self.optimiser_confidence = optimiser_confidence
        self.basinhopping_confidence = basinhopping_confidence

        self.initial = kwargs['initial']
        self.step_noise = kwargs['step_noise']
        self.bounds = bounds
        
        self.debug = debug

        if len(self.initial) != len(self.step_noise):
            raise Exception(
                "Step noise and initial conditions must be equal length."
            )

        self.result = self.run_basinhopping()

    def step_function(self, optimiser_input):
        for i, noise in enumerate(self.step_noise):
            optimiser_input[i] += np.random.normal(scale=noise)

        return optimiser_input

    def callback_function(self,
                          optimiser_output,
                          minimise_function_result,
                          was_accepted):
        if type(self.debug) is not None:
            self.debug(optimiser_output)
        
        if not(was_accepted):
            return

        if self.current_success_number == 0:
            # First result
            self.successful_results[0] = minimise_function_result
            self.current_success_number = 1

        elif (minimise_function_result >=
              np.nanmin(self.successful_results) + self.basinhopping_confidence):
            # Reject result
            0

        elif (minimise_function_result >=
              np.nanmin(self.successful_results) - self.basinhopping_confidence):
            # Agreeing result
            self.successful_results[
                self.current_success_number
            ] = minimise_function_result

            self.current_success_number += 1

        elif (minimise_function_result <
              np.nanmin(self.successful_results) - self.basinhopping_confidence):
            # New result
            self.successful_results[0] = minimise_function_result
            self.current_success_number = 1

        if self.current_success_number >= self.n:
            return True

    def run_basinhopping(self):
        self.successful_results = np.empty(self.n)
        self.successful_results[:] = np.nan
        self.current_success_number = 0

        minimizer_config = {
            "method": 'L-BFGS-B',
            "options": {'gtol': self.optimiser_confidence},
            "bounds": self.bounds
        }

        output = basinhopping(
            self.to_minimise,
            self.initial,
            niter=1000,
            minimizer_kwargs=minimizer_config,
            take_step=self.step_function,
            callback=self.callback_function
        )

        return output.x
