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
"""Supporting classes for recursively computing diurnal energy on address blocks.

"""
import unittest
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import math
from energy import power_ratio

################################################################
# Some helper classes

OneDay = 86400          # seconds per day

class TimeBucket():
  """Tag tests by time, rounded into buckets.

  The bucketsize must be multiples of minutes, specified in seconds,
  and divide into 24 hours.  (Required by other parts of the
  algorithm).
  """

  def __init__(self, bucketsize):
    """Create the class with the specified bucket size.
    """
    assert(bucketsize %60 == 0) # Manually tested once
    assert(OneDay % bucketsize == 0) # Manually tested once
    self.bucketsize = bucketsize

  def bucket(self,time):
    """Convert seconds to a bucket name string: T00:00 - T23:59 """

    bucket = ((time / self.bucketsize) * self.bucketsize) % OneDay
    return "T%02d:%02d"%(bucket/3600, (bucket/60)%60)

  def all(self):
    """Return a list of all bucket names."""

    return([self.bucket(i) for i in range(0, OneDay, self.bucketsize)])

################################################################
class NetBlock():
  """Process and score lists of tests

  A netblock is a DataFrame where:
    rows correspond to tests (NDT etc);
    columns correspond to Web100 vars and computed scores for each row.

  Each netblock only contains tests (rows) selected by some outer loop.
  """

  def __init__(self, timebucket):
    """
    The size of the timebucket is an invariant
    """
    self.tb = TimeBucket(timebucket)

  def parse(self, itr, cannon, cols=None):
    """Parse, canonicalize and score imported data into a NetBlock DataFrame

    itr can be any iterator (accepted for DataFrame(itr, ...)
    cannon(row) is a canonicalization operation applied to each row
    cols is an optional column specifier needed
        if list(itr) does not return s suitable list of columns.

    cannon(row, tb) should end with:
      row[tb.bucket(t)] = v # Where t is the time and v the value of one measurement

    """
    if not cols:
      cols = list(itr)
    cols = self.tb.all() + cols
    self.data = DataFrame(itr, columns = cols)
    self.data = self.data.apply(cannon,
                    axis = 'columns',
                    raw = True,
                    args = [self.tb])
    return self

  def energy(self, harmonics=1):
    """Compute the energy of this Netblock.

    Returns a 2tuple:
    energy at 1/(24hr) and selected harmonics
    total energy

    NB: this may be superseded by a future summary function.
    """

    timeseries = [self.data[tt].sum() for tt in self.tb.all()]
    timeseries = [0.0 if np.isnan(tv) else tv for tv in timeseries]
    return power_ratio(timeseries, len(self.tb.all()), harmonics)

################################################################
class ScoreFrame(DataFrame):
  """Manipulate netblocks and scores

  A ScoreFrame is a DataFrame where:
    Rows correspond to netblocks with attributes and scores
    Columns include scores, address masks, and the netblocks themselves

  """
  def __init__(self):
    self.rank = 0

  def process(self):

    # Bump netmask
    # cut block on first row
    If rest is empty: loop
    score child
    score remainder
    if interesting:
      push child
      push remainder



  def fork(self, pred):
    # Filter on predicate
    # delete child rows from parent
    return (child)

  def annotate(self):
    # compute scores

  def push(self, list)
  def pop(list)

################################################################
# built in testers

class TestNetblock(unittest.TestCase):
  def test_time2bucket(self):
    tb = TimeBucket(300)
    self.assertEqual(tb.bucket(0), "T00:00")
    self.assertEqual(tb.bucket(1), "T00:00")
    self.assertEqual(tb.bucket(299), "T00:00")
    self.assertEqual(tb.bucket(300), "T00:05")
    self.assertEqual(tb.bucket(3600), "T01:00")
    self.assertEqual(tb.bucket(24*60*60-1), "T23:55")
    self.assertEqual(tb.bucket(24*60*60), "T00:00")
    all = tb.all()
    self.assertEqual(all[0], "T00:00")
    self.assertEqual(all[1], "T00:05")
    self.assertEqual(len(all), OneDay / 300)

  def test_netblock(self):
    def smear(seq):
      """Spread a time series as one point per row in a NetBlock."""
      return(pd.DataFrame({
          "Value": pd.Series(seq),
          "Time" : pd.Series(range(0, OneDay,OneDay/len(seq)))
      }))

    def canonical(row, tb):
      row[tb.bucket(row["Time"])] = row["Value"]
      return (row)

    def approxEQ(c, p, tp):  # Copied from TestEnergy in energy.py
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
    ss2 = size*size/4 # TODO(mattmathis) why /4?
    nb = NetBlock(OneDay/size)
    # test cases from TestEnergy in energy.py
    sinewave1 = [math.sin((math.pi*2*t)/size) for t in range(size)]
    sinewave2 = [math.sin((math.pi*4*t)/size) for t in range(size)]
    self.assertTrue(approxEQ(nb.parse(smear(sinewave1), canonical).energy(), ss2, ss2))
    self.assertTrue(approxEQ(nb.parse(smear(sinewave2), canonical).energy(), 0, ss2))
    self.assertTrue(approxEQ(nb.parse(smear(sinewave2), canonical).energy(2), ss2, ss2))

  def test_scoreframe(self):
    pass

if __name__ == "__main__":
  unittest.main()
