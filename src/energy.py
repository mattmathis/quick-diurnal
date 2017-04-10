#!/usr/bin/env python
# Copyright 2017 The MLab quick diurnal searcher
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Library to compute the total energy at 1/(24h) and harmonics.

Also computes the total energy.  The ratio of these 2 energies is a
scalar signal of how likely there is some networking behavior related
to the diurnal cycles of human behavior.  Note that this is expected
for some parameters, such as traffic volumes.  However it is not
expected for performance signals that arising from edge networks.

This is intended to be used as a library.  If run directly, it does a unit test.
"""

import datetime
import numpy
import math

def power_ratio(timeseries, period, harmonics=1):
  """Compute the energy at the period and harmonics

  Args:
    timeseries: A list of uniformly spaced data points.
    period: The number of bins per cycle.
    harmonics: Number of harmonics to include in the energy calculation

  This works for both long time series (exact multiples of period) and
  wrapped time series (the time series has exactly period bins).

  The period must be a multiple of all included harmonics.

  Returns a 2tuple:
    Total power at 1/period and harmonics
    Total power in the signal

  The ratio of these is the fraction of the signal energy at 1/period.
  We expose the numerator and denominator separately to make it easier
  for the caller to implement gradient assent algorithms.
  """

  if len(timeseries)%period != 0:
    return False
  spectrum = numpy.fft.rfft(timeseries)
  power = [x.real*x.real + x.imag*x.imag for x in spectrum]
  hpower = 0.0
  for h in range(1, harmonics+1):
    hpower = hpower + power[h*len(timeseries)/period]
  return ((hpower, numpy.sum(power)))


import unittest

class TestEnergy(unittest.TestCase):
  def test_power_ratio(self):
    def approxEQ(c, p, tp):
      """Result tester for power_ratio()
      c: Two tuple from power_ratio() above
      p: Computed power at 1/(20h) and harmonics
      tp: Computed total power

      Returns True if both values agree within lim
      """
      lim = 0.00001
      cp, ctp = c
      return ((cp-p)<lim) and ((ctp-tp)<lim)
    size = 24*12
    uniform = [1.0 for t in range(size)]
    sinewave1 = [math.sin((math.pi*2*t)/size) for t in range(size)]
    sinewave2 = [math.sin((math.pi*4*t)/size) for t in range(size)]
    impulse = [0.0 for t in range(size)]
    impulse[0] = 1.0
    # Error expected
    self.assertFalse(power_ratio(sinewave1, size-1))
    self.assertTrue(approxEQ(power_ratio(uniform, size), 0.0, size*size))
    ss2=size*size/2 # expected power for sine signals
    self.assertTrue(approxEQ(power_ratio(sinewave1, size), ss2, ss2))
    self.assertTrue(approxEQ(power_ratio(sinewave2, size), 0.0, ss2))
    self.assertTrue(approxEQ(power_ratio(sinewave2, size, 1), 0.0, ss2))
    self.assertTrue(approxEQ(power_ratio(sinewave2, size, 3), ss2, ss2))
    self.assertTrue(approxEQ(power_ratio(sinewave2, size, 3), ss2, ss2))
    self.assertTrue(approxEQ(power_ratio(sinewave2, size, 4), ss2, ss2))
    ssi=size/2+1 # expected impulse power
    self.assertTrue(approxEQ(power_ratio(impulse, size, 1), 1.0, ssi))
    self.assertTrue(approxEQ(power_ratio(impulse, size, 2), 2.0, ssi))
    self.assertTrue(approxEQ(power_ratio(impulse, size, 3), 3.0, ssi))
    self.assertTrue(approxEQ(power_ratio(impulse, size, 4), 4.0, ssi))


if __name__ == "__main__":
  unittest.main()
