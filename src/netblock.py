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
import random

################################################################
# Some helper classes and functions

################ Time Buckets
OneDay = 86400          # seconds per day
OneWeek = 7*OneDay      # seconds per week
FirstSunday = 3*OneDay  # Unix time 'Sun, 04 Jan 1970 00:00:00 +0000'

class TimeBucket():
  """Tag tests by time, rounded into buckets.

  The bucket size must be multiples of minutes, specified in seconds,
  and divide into 24 hours.  (Required by other parts of the
  algorithm).
  """

  def __init__(self, bucketsize):
    """Create the class with the specified bucket size.
    """
    assert(bucketsize %60 == 0) # Manually tested once
    assert(OneDay % bucketsize == 0) # Manually tested once
    self.bucketsize = bucketsize
    self.bucketcount = OneDay / bucketsize
    self.all = [self.bucket(i) for i in range(0, OneDay, self.bucketsize)]

  def bucket(self,time):
    """Convert seconds to a bucket name string: T00:00 - T23:59 """

    bucket = (int(time / self.bucketsize) * self.bucketsize) % OneDay
    return "T%02d:%02d"%(bucket/3600, (bucket/60)%60)

################ IP address and subnet tools
from netaddr import IPAddress

def inet_addr(s):
  """Convert IP address string xxx.xxx.xxx.xxx to int"""
  return int(IPAddress(s))

def inet_ntoa(i):
  """Convert int to address string xxx.xxx.xxx.xxx"""
  return str(IPAddress(i))

class SubNet():
  """Primitives for creating an manipulating subnet masks"""

  def __init__(self, width, address):
    """Create a subnet mask from an address at a specified width.

    NB: does not support IPv6
    """
    self.width = width
    self.mask = (2 ** 32) - (2 ** (32 - width))
    self.prefix = self.mask & address
    self.last = (2 ** (32 - width)) - 1 + self.prefix

  def match(self, address):
    """Check if an address matches a subnet

    NB: Match may be expanded inline for pandas iterators
    """
    return (self.mask & address) == self.prefix

  def invert(self):
    """Toggle the last bit of the prefix"""
    prefix = self.prefix ^ (2 ** (32 - self.width))
    return SubNet(self.width, prefix)

  def str(self):
    """Display conventional prefix/len notation"""
    return "%s/%d"%(inet_ntoa(self.prefix), self.width)

  def sstr(self):
    """Display naked prefix (no len).  Drop trailing zeros"""
    s = inet_ntoa(self.prefix)
    while s[-2:] == '.0':
      s = s[:-2]
    return s

  def vstr(self):
    """Verbose display: prefix/len (netmask) - last"""
    return "%s/%d (%s) - %s"%(inet_ntoa(self.prefix),
                              self.width,
                              inet_ntoa(self.mask),
                              inet_ntoa(self.last))

################ Calendar queue (aka heap queue)
import heapq
def hpush(heap, item):
  """Queue ordered by item.rank"""
  heapq.heappush(heap, (item.rank, item))

def hpop(heap):
  """Dequeue, discard rank"""
  rank, item = heapq.heappop(heap)
  return item

################################################################
class NetBlock():
  """Process and score lists of tests

  A NetBlock represents a group of tests sharing attributes.

  It includes parameters describing the shared attributes:
    .subnet - width, netmask and prefix identifying a subnet
    .week - [FUTURE] week start time

  It includes a DataFrame .data where:
    rows correspond to tests (NDT etc);
    columns correspond to Web100 vars and computed per test scores.

  And computed results
    .energy - computed energies for this NetBlock
    .rank - Priority for recursive processing

  TODO: It might be more efficient to use panels within one netframe.
  """
  # Invariant after first instantiation
  timebucket = None
  TB = None
  TBall = []
  canonF = None
  energyF = None
  # Shared state
  todo = [] # Priority Queue
  done = []

  def __init__(self, timebucket=0, canon = None, energy = None, rank = None):
    """
    First invocation require additional arguments:
    timebucket - Time quantization size in seconds (e.g. 300)
    canon(row) - a canonicalization operation applied to each row
    energy - energy calculation for the NetBlock
    rank - optional rank score calculation (processing priority)

    canon(row) should include:
      row["clientIP"] = inet_addr(row["client_ip_v4"])
      row["Time"] = int(row["start_time"]/1000000)
      row["Week"] = int((row["Time"]-FirstSunday)/OneWeek)*OneWeek+FirstSunday
      row["Value"] = some_value
      row[tb.bucket(row["Time"])] = row["Value"]
    NB: clientIP, Time, Week and Value, are accessed elsewhere

    """
    if timebucket != 0:
      NetBlock.TB = TimeBucket(timebucket)
      NetBlock.TBall = NetBlock.TB.all
      NetBlock.canonF = canon
      NetBlock.energyF = energy
      NetBlock.rankF = rank     # optional
      if not (timebucket and canon and energy):
        print "Warning: missing argument(s) on first NetBlock()"

  def parse(self, itr, cols=None, downsample=1):
    """Parse, canonicalize and score imported data into a NetBlock DataFrame

    itr can be any iterator accepted for DataFrame(itr, ...)

    cols is an optional column specifier (defaults to list(itr)).

    TODO(mattmathis) multiple parse calls should append data.
    """
    if not cols:
      cols = list(itr)
    cols = NetBlock.TBall + cols
    self.data = pd.DataFrame(itr, columns = cols)
    if downsample > 1:
      print "Warning %d:1 random down sampling"%downsample
      mask=pd.Series(np.random.randint(0, downsample, len(self.data)) == 0)
      self.data = self.data[mask]
    self.data = self.data.apply(self.canonF,
                                axis = 'columns',
                                raw = True,
                                args = [self.TB])
    return self

  def first_row(self):
    assert (len(self.data) > 0)
    return (self.data.iloc[0])

  def fork_block(self, sn=None, rowmask=[]):
    """Create a new NetBlock and compute initial scores
    from:
       self - the parent
       sn - a new subnet, or
       rowmask - an arbitrary row mask
    """
    child = NetBlock()
    if sn == None:
      sn = self.subnet
    child.subnet = sn
    if len(rowmask) == 0:
      rowmask = self.data.clientIP.apply(sn.match)
    child.data = DataFrame.reindex(self.data[rowmask])
    if len(child.data) > 0:
      child.energy = child.energyF()
      if child.rankF:
        child.rank = child.rankF()
    return child

  def process(self, go_deeper):
    """Recursively split and queue NetBlock for processing

    go_deeper() controls the recursion
    """
    neww = self.subnet.width+1
    subA = SubNet(neww, self.first_row().clientIP)
    subB = subA.invert()
    blockB = self.fork_block(subB)
    while len(blockB.data) == 0:
      neww = neww + 1
      if neww > 24:
        break
      subA = SubNet(neww, self.first_row().clientIP)
      subB = subA.invert()
      blockB = self.fork_block(subB)
    else:
      blockA = self.fork_block(subA)
      if go_deeper(self, blockA, blockB):
        hpush(NetBlock.todo, blockA)
        hpush(NetBlock.todo, blockB)
        return
    self.subnet.width = neww - 1
    hpush(NetBlock.done, self)

  def norm_spectra(self, harmonics=1, shuffle=0, power=False):
    """Compute the normalized frequency spectra

    Returns summary statistics of the spectra in a dictionary.

    harmonics - How many harmonics of (1/24h) to include
    shuffle - shuffle the data to destroy all diurnal signals.
       shuffle=0: Use fast algorithm
       shuffle=1: Use isomorphic slow algorithm
       shuffle=2: Shuffle the raw data
            (For calibration experiments.)
    power - compute total signal power at multiple stages

    Returns:
    nrows: Number of rows in the original (raw) set
    rawsum: Sum of raw values
    mean: Mean of the raw values
    nan: Number of time bins with no results
    mag: Series of harmonics
    sum24: Sum of selected harmonics
    tsig: Sum of all frequencies (excludes mean)
    ratio: sum24/tsig
    nratio: sum24/tsig / harmonics/size  E(nratio) == 1.0 for random data
    if power is True, also return:
      rawpower - Power directly computed from the input data
      bktpower - power computed after reduction to one diurnal series
      spectrapower - computed from the coefficients
      NB: In theory all three should be the same but since we have non-periodic
      data but only periodic spectral frequencies, the energies do not match.

    """
    size = NetBlock.TB.bucketcount
    scale1 = 2.0/size # scale factor for normalizing signal
    nrows = self.data["Value"].count()
    if nrows < 1:
      return {'nrows':nrows}
    rawsum = self.data["Value"].sum()
    mean = rawsum / nrows
    if shuffle:
      timeseries = self.data["Time"].reset_index(drop=True)
      dataseries = self.data["Value"].reset_index(drop=True)
      if shuffle > 1:
        random.shuffle(dataseries)
        dataseries = dataseries.reset_index(drop=True)
      assert len(dataseries) == len(timeseries)
      bucketseries = Series((timeseries % OneDay) / self.TB.bucketsize, dtype=int)
      timebuckets = Series([np.nan for i in range(size)])
      for i in range(size):
        tmp = dataseries[bucketseries == i]
        timebuckets[i] = tmp.sum() - mean*tmp.count()
    else:
      timebuckets = Series([self.data[tt].sum() - mean*self.data[tt].count() \
                  for tt in  NetBlock.TBall])
    nan = len(timebuckets) - timebuckets.count()
    if nan:
      timebuckets = [0.0 if np.isnan(tv) else tv for tv in timebuckets]
    spectrum = np.fft.rfft(timebuckets)
    magnitude = Series([abs(x)*scale1 for x in spectrum])
    # phase = Series([phase(x) for x in spectrum]) - Not useful here
    assert magnitude[0] < 0.0001 # Confirm proper pre bias
    tsig = magnitude.sum()        # Total signal (excludes mean)
    sum24 = magnitude[1:harmonics+1].sum()
    nratio = sum24/var
    try:
      ratio = sum24/tsig
    except:
      # failed to fill all NaNs or other failures
      ratio = np.nan
    ret = {'nrows':nrows, 'rawsum':rawsum, 'mean':mean, 'nan':nan, \
            'mag':magnitude, 'sum24':sum24, 'tsig':tsig, 'var':var, 'ratio':ratio, 'nratio':nratio}
    if power:
      scale2 = size/2.0
      rawpower = sum([x*x for x in self.data["Value"]])
      acpower = sum([x*x for x in timebuckets])
      bktpower = acpower + mean*mean*size
      spectrapower = sum([x*x for x in magnitude])*scale2 + mean*mean*size
      pratio = spectrapower/bktpower
      ret.update({'rawpower':rawpower, 'bktpower':bktpower, 'spectrapower':spectrapower, 'pratio':pratio,})
    return ret

################################################################
# built in testers

# Helpers first
def test_energy_canon(nb, row, tb):
  """Test helper to do minimal result canonicalization"""
  row[tb.bucket(row["Time"])] = row["Value"]
  return row

def test_subnet_canon(nb, row, tb):
  """Test helper to do minimal result canonicalization"""
  row["Value"] = np.nan
  row["clientIP"] = inet_addr(row["client_ip_v4"])
  return row

def test_parse_row(nb, row, tb):
  """Minimal parser for real data"""
  row["clientIP"] = inet_addr(row["client_ip_v4"])
  row["Value"] = row["download_mbps"]
  row[tb.bucket(row["Time"])] = row["Value"]
  return row

def smear(seq):
  """Spread a time series as one point per row in a NetBlock."""
  return(pd.DataFrame({
      "Value": pd.Series(seq),
      "Time" : pd.Series(range(0, OneDay,OneDay/len(seq))),
      }))

def CheckVals(D, argdict):
  """Check one dictionary against another

  argdict is **kwargs in an outer context.
  """
  lim = 0.00001
  r = True
  if argdict is not None:
    for key, val in argdict.iteritems():
      if not key in D:
        print "Key %s missing"%key
        r = False
      elif val is None:
        print "Show key %s is %s"%(key, D[key])
      elif abs(D[key] - val) > lim:
        print "Key %s %s != %s"%(key, D[key], val)
        r = False
  return (r)

class Test_Netblock(unittest.TestCase):

################
  def test_time2bucket(self):
    tb = TimeBucket(300)
    self.assertEqual(tb.bucket(0), "T00:00")
    self.assertEqual(tb.bucket(1), "T00:00")
    self.assertEqual(tb.bucket(299), "T00:00")
    self.assertEqual(tb.bucket(300), "T00:05")
    self.assertEqual(tb.bucket(3600), "T01:00")
    self.assertEqual(tb.bucket(24*60*60-1), "T23:55")
    self.assertEqual(tb.bucket(24*60*60), "T00:00")
    all = tb.all
    self.assertEqual(all[0], "T00:00")
    self.assertEqual(all[1], "T00:05")
    self.assertEqual(len(all), OneDay / 300)

################
  def test_IPaddr(self):
    self.assertEqual(inet_addr('192.168.4.54'), 3232236598)
    self.assertEqual(inet_ntoa(3232236598), '192.168.4.54')
    sn = SubNet(16, inet_addr('192.168.4.54'))
    self.assertEqual(sn.mask, inet_addr('255.255.0.0'))
    self.assertEqual(sn.prefix, inet_addr('192.168.0.0'))
    self.assertEqual(sn.invert().prefix, inet_addr('192.169.0.0'))
    self.assertTrue(sn.match(inet_addr('192.168.1.2')))
    self.assertFalse(sn.match(inet_addr('192.169.1.2')))

################
  def AssertSectraVals(self, nb, seq, harm, **kwargs):
    nb.parse(smear(seq))
    self.assertTrue(CheckVals(nb.norm_spectra(harm, power=True), kwargs))

  def test_netblock_spectra(self):
    """Test spectra using known functions."""
    size = 24*12
    nb = NetBlock(OneDay/size, test_energy_canon, True, True)
    uniform = Series([1.0 for t in range(size)])
    self.AssertSectraVals(nb, uniform, 1, \
                          nrows=size, rawsum=size, mean=1.0, nan=0, sum24=0.0, tsig=0.0)
    sinewave1 = Series([math.sin((math.pi*2*t)/size) for t in range(size)])
    self.AssertSectraVals(nb, sinewave1, 1, \
                          rawpower=size/2.0, bktpower=size/2.0, spectrapower=size/2.0, \
                          nrows=size, rawsum=0.0, mean=0.0, nan=0, sum24=1.0, tsig=1.0, ratio=1.0)
    sinewave2 = Series([math.sin((math.pi*4*t)/size) for t in range(size)])
    self.AssertSectraVals(nb, sinewave2, 1, sum24=0.0, tsig=1.0)
    self.AssertSectraVals(nb, sinewave2, 2, sum24=1.0, tsig=1.0)
    self.AssertSectraVals(nb, sinewave2, 3, sum24=1.0, tsig=1.0)
    impulse = Series([0.0 for t in range(size)])
    impulse[0] = 1.0
    self.AssertSectraVals(nb, impulse, 1, rawpower=1.0, bktpower=1.0, spectrapower=1.0034722)
    self.AssertSectraVals(nb, impulse, 1, sum24=2.0*1/size, tsig=1.0, ratio=2.0*1/size)
    self.AssertSectraVals(nb, impulse, 2, sum24=2.0*2/size, tsig=1.0, ratio=2.0*2/size)
    self.AssertSectraVals(nb, impulse, 3, sum24=2.0*3/size, tsig=1.0, ratio=2.0*3/size)
    self.AssertSectraVals(nb, impulse, 4, sum24=2.0*4/size, tsig=1.0, ratio=2.0*4/size)
    inan =  Series([np.nan for t in range(size)])
    inan[0] = 1.0
    self.AssertSectraVals(nb, inan, 1, \
                          nrows=1, rawsum=1.0, mean=1.0, nan=size-1, sum24=0.0, tsig=0.0)
    inan =  Series([np.nan for t in range(size)])
    inan[0] = 1.0
    inan[1] = -1.0
    self.AssertSectraVals(nb, inan, 1, \
                          nrows=2, rawsum=0.0, mean=0.0, nan=size-2)
    # TODO Understand and model sum24 and tsig for the above case (and add more)
    # TODO (Monte Carlo) Proof that for random data E(ratio) = num_harmonics/size
    # TODO model for Var(Ratio) as a function of Var(data)

  def Xtest_netblock_specta_properties(self):
    """Confirm spectra properties on authentic data.

    This test does not work well enough to be useful.

    We need to understand how shuffling the data effects spectra variance.
    """
    return # @@@@@@ FIXME

    buckets = 24*1

    testdata = NetBlock(OneDay/buckets, test_parse_row)
    testdata.parse(pd.read_csv(open("testdata.csv")), downsample=1)

    fmt = "{mean} {rawpower} {bktpower} {spectrapower} {pratio} {sum24} {tsig} {ratio} {nratio}"
    print "    "+fmt
    print "s=0 "+fmt.format(**testdata.norm_spectra(shuffle=0, power=True))
    print "s=1 "+fmt.format(**testdata.norm_spectra(shuffle=1, power=True))
    for i in range(10):
      s=testdata.norm_spectra(shuffle=2, power=True)
      print "s=2 "+fmt.format(**s)

################
  def test_netblock_subnets(self):
    def test_rank(nb):
      return nb.first_row().score
    def always_true(nb, *rest):
      return True
    def always_false(nb, *rest):
      return False

    rawdata = pd.DataFrame({
        'client_ip_v4': ["10.1.1.1", "10.2.1.1", "10.2.1.2", "11.4.1.1"],
        'score': [ 2, 1, 1, 1],
        })

    parent = NetBlock(OneDay/8,
                  test_subnet_canon,
                  NetBlock.norm_spectra,
                  test_rank)
    # All rawdata got parsed
    parent.parse(rawdata)
    self.assertEqual(len(parent.data), 4)
    # fork_block got the right number of rows
    sn = SubNet(8, parent.first_row().clientIP)
    child = parent.fork_block(sn)
    self.assertEqual(len(child.data), 3)
    self.assertEqual(len(parent.data), 4)
    # Confirm non-recursion case
    child.process(always_false)
    self.assertEqual(len(NetBlock.done), 1)
    self.assertEqual(len(NetBlock.todo), 0)
    # parent is done and has a minimal subnet
    parent2 = hpop(NetBlock.done)
    self.assertEqual(parent2.subnet.str(), "10.0.0.0/14")
    self.assertEqual(len(NetBlock.done), 0)
    # Confirm recursion case
    child.process(always_true)
    self.assertEqual(len(NetBlock.done), 0)
    self.assertEqual(len(NetBlock.todo), 2)
    # Two children in the proper order and subnets
    child2 = hpop(NetBlock.todo)
    child3 = hpop(NetBlock.todo)
    self.assertEqual(len(NetBlock.todo), 0)
    self.assertEqual(child2.subnet.str(), "10.2.0.0/15")
    self.assertEqual(child3.subnet.str(), "10.0.0.0/15")

if __name__ == "__main__":
  unittest.main()
