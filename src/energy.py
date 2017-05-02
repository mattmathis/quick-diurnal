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
expected for performance signals that arise from edge networks.

This is intended to be used as a library.  If run directly, it does a unit test.

This entire module has been superceeded by netblock.py::norm_spectra()

"""

import datetime
from pandas import DataFrame, Series
import numpy
import math

def power_ratio(timeseries, period, harmonics=1):
  """Compute the energy at the period and harmonics

  OBSOLETE See netblock.py::norm_spectra()

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

  print "energy.py::power_ratio is obsolete"
  if len(timeseries)%period != 0:
    return False
  spectrum = numpy.fft.rfft(timeseries)
  power = [abs(x) for x in spectrum]
  # power = [x.real*x.real + x.imag*x.imag for x in spectrum]
  hpower = 0.0
  for h in range(1, harmonics+1):
    hpower = hpower + power[h*len(timeseries)/period]
  return ((hpower, numpy.sum(power)))

def new_power(timeseries, period, harmonics=1):
  """Compute the energy at the period and harmonics

  OBSOLETE See netblock.py::norm_spectra()

  NOTE THIS IS NOT (QUITE) THE RIGHT APPROACH.


  Args:
    timeseries: A list of uniformly spaced data points.
    period: The number of bins per cycle.
    harmonics: Number of harmonics to include in the energy calculation

  This works for both long time series (exact multiples of period) and
  wrapped time series (the time series has exactly period bins).

  The period must be a multiple of all included harmonics.

  Returns a dictionary:
  e24 - energy at 1/(24hr) and selected harmonics
  te - total energy
  ra = e24/te
  nan - Number of time bins for which values had to be interpolated

  Ra, the ratio of e24 and te is the fraction of the signal energy at 1/period.
  We expose the numerator and denominator separately to make it easier
  for the caller to implement gradient assent algorithms.
  """
  print "energy.py::new_power is obsolete"
  if len(timeseries)%period != 0:
    return False

  mean = timeseries.mean()
  nan = len(timeseries) - timeseries.count()
  if nan:
    timeseries = [mean if numpy.isnan(tv) else tv for tv in timeseries]
  spectrum = numpy.fft.rfft(timeseries)
  power = Series([abs(x) for x in spectrum])
  dc = power[0] - nan * mean    # DC part of the power
  power[0] = dc
  te = power.sum()              # Total energy
  e24 = 0.0                     # Energy at harmonics of 1/24h
  for h in range(1, harmonics+1):
    e24 += power[h*len(timeseries)/period]
  try:
    ra = e24/te
  except:
    # failed to fill all NaNs or other failure
    ra = float('nan')
  return { 'e24':e24, 'te':te, 'ra':ra, 'nan':nan, 'dc':dc, 'ac':te-dc}

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
      if (abs(cp-p)<lim) and (abs(ctp-tp)<lim):
        return True
      print " %s != %s or %s != %s"%(cp, p, ctp, tp)
      return False
    size = 24*12
    uniform = [1.0 for t in range(size)]
    sinewave1 = [math.sin((math.pi*2*t)/size) for t in range(size)]
    sinewave2 = [math.sin((math.pi*4*t)/size) for t in range(size)]
    impulse = [0.0 for t in range(size)]
    impulse[0] = 1.0
    sqrt2 = math.sqrt(2)
    # Error expected
    self.assertFalse(power_ratio(sinewave1, size-1))
    self.assertTrue(approxEQ(power_ratio(uniform, size), 0.0, size))
    ss2=size/2 # expected power for sine signals
    self.assertTrue(approxEQ(power_ratio(sinewave1, size), ss2, ss2))
    self.assertTrue(approxEQ(power_ratio(sinewave2, size), 0.0, ss2))
    self.assertTrue(approxEQ(power_ratio(sinewave2, size, 1), 0.0, ss2))
    self.assertTrue(approxEQ(power_ratio(sinewave2, size, 3), ss2, ss2))
    self.assertTrue(approxEQ(power_ratio(sinewave2, size, 3), ss2, ss2))
    self.assertTrue(approxEQ(power_ratio(sinewave2, size, 4), ss2, ss2))
    ssi=ss2+1 # expected impulse power
    self.assertTrue(approxEQ(power_ratio(impulse, size, 1), 1.0, ssi))
    self.assertTrue(approxEQ(power_ratio(impulse, size, 2), 2.0, ssi))
    self.assertTrue(approxEQ(power_ratio(impulse, size, 3), 3.0, ssi))
    self.assertTrue(approxEQ(power_ratio(impulse, size, 4), 4.0, ssi))

  def CheckVals(self, R, e24=None, te=None, ra=None, nan=None):
    lim = 0.00001
    r = True
    if e24 != None and abs(R['e24']-e24) > lim:
      print "Fail e24 %s != %s "%(R['e24'], e24)
      r = False
    if te != None and abs(R['te']-te) > lim:
      print "Fail te %s != %s "%(R['te'], te)
      r = False
    if ra != None and abs(R['ra']-ra) > lim:
      print "Fail ra %s != %s "%(R['ra'], ra)
      r = False
    if nan != None and abs(R['nan']-nan) > lim:
      print "Fail nan %s != %s "%(R['nan'], nan)
      r = False
    return (r)

  def AssertPowerVals(self, ts, harm=1, e24=None, te=None, ra=None, nan=None):
    self.assertTrue(self.CheckVals(new_power(ts, len(ts), harm), e24, te, ra, nan))

  def test_new_power(self):
    size = 24*12
    uniform = Series([1.0 for t in range(size)])
    sinewave1 = Series([math.sin((math.pi*2*t)/size) for t in range(size)])
    sinewave2 = Series([math.sin((math.pi*4*t)/size) for t in range(size)])
    impulse = Series([0.0 for t in range(size)])
    impulse[0] = 1.0
    inan =  Series([float('nan') for t in range(size)])
    inan[0] = 1.0
    sqrt2 = math.sqrt(2.0)
    self.assertFalse(new_power(sinewave1, size-1))
    self.AssertPowerVals(uniform, 1, 0.0, size)
    ss2=size/2 # expected power for sine signals
    self.AssertPowerVals(sinewave1, 1, ss2, ss2)
    self.AssertPowerVals(sinewave2, 1, 0.0, ss2)
    self.AssertPowerVals(sinewave2, 1, e24=0.0, te=ss2)
    self.AssertPowerVals(sinewave2, 2, ss2, ss2)
    self.AssertPowerVals(sinewave2, 3, ss2, ss2)
    self.AssertPowerVals(sinewave2, 4, ss2, ss2)
    ssi=ss2+1 # expected impulse power
    self.AssertPowerVals(impulse, 1, 1.0, ssi)
    self.AssertPowerVals(impulse, 2, 2.0, ssi)
    self.AssertPowerVals(impulse, 3, 3.0, ssi)
    self.AssertPowerVals(impulse, 4, 4.0, ssi)
    # This has no signals
    self.AssertPowerVals(inan, 1, 0.0, 1.0, 0.0, size-1)

if __name__ == "__main__":
  unittest.main()
