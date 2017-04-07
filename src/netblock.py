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
from pandas import DataFrame, Series
import pandas as pd

################################################################
# Some helper functions

OneDay = 86400          # seconds per day
BucketSize = 3600       # seconds per bucket

def time2bucket(time):
  """ convert seconds to a bucket name string: T00:00 - T23:59 """
  time = time % OneDay
  return "T%02d:%02d"%(time/3600, (time/60)%60)

################################################################
class NetBlock(DataFrame):
  """Process and score lists of tests

  A netblock is a DataFrame where:
    rows correspond to tests (NDT etc);
    columns correspond to Web100 vars and computed scores for each row.

  """

  def __init__(self):
    pass

  def parse(self, itr):
    self.data = DataFrame(itr)
    # do input cononicalization
    # add diurnal columns

  def score(self):
    # sum diurnal columns into diurnal bins
    # Compute FFT
    # save energy
    # compute and save rank_score
    # Returns a score tupple
    pass

################################################################
class ScoreFrame(DataFrame):
  """Manipulate netblocks and scores

  A ScoreFrame is a DataFrame where:
    Rows correspond to scored netblocks
    Columns include scores, address masks, and the netblocks themselves

  """
  def __init__(self):
    self.rank = 0

################################################################
# built in testers

def test_time2bucket():
  assert (time2bucket(0) == "T00:00")
  assert (time2bucket(300) == "T00:05")
  assert (time2bucket(3600) == "T01:00")
  assert (time2bucket(24*60*60-1) == "T23:59")
  assert (time2bucket(24*60*60) == "T00:00")

def test_netblock():
  pass

def test_scoreframe():
  pass

if __name__ == "__main__":
  test_time2bucket()
  test_netblock()
  test_scoreframe()
  print "All pass"
