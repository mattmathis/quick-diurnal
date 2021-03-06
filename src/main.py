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
"""Quick Signal Searcher



"""
import sys
from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import math
import netblock as NB
import energy
import csv

ALLCOLS = ['server_ip_v4', 'client_ip_v4', 'start_time', 'Duration',
           'download_mbps', 'min_rtt', 'avg_rtt',
           'retran_per_DataSegsOut', 'retran_per_CongSignals']

def parse_row(nb, row, tb):
  row["clientIP"] = NB.inet_addr(row["client_ip_v4"])
  row[tb.bucket(row["start_time"])] = row["download_mbps"]
#  row[tb.bucket(row["start_time"])] = row["avg_rtt"]
  return row

def energy_mean_nan(nb, harmonics=2, method='ffill'):
  """Compute the energy of this NetBlock.

  Returns a dictionary:
  e24 - energy at 1/(24hr) and selected harmonics
  te - total energy
  ra = e24/te
  nrows - Number of datapoints in the sample
  nan - Number of time bins for which values had to be interpolated
  """

  # Note that "mean" is not automatically correct.  Consider median, others
  timeseries = Series([nb.data[tt].mean() for tt in  nb.TBall])
  nan = nb.TB.bucketcount - timeseries.count()
  if method:
    timeseries = timeseries.fillna(method=method)
    # TODO (mattmathis) exlore limit= options
  else:
    # zero fill is only correct for energy algebra
    timeseries = [0.0 if np.isnan(tv) else tv for tv in timeseries]
  e24, te = energy.power_ratio(timeseries, len(timeseries), harmonics)
  try:
    ra = e24/te
  except:
    # failed to fill all NaNs or other failure
    ra = float('nan')
  return { 'e24':e24, 'te':te, 'ra':ra, "nrows":len(nb.data), 'nan':nan }

def rank(nb):
  """Provide a preliminary estimate of interest"""
  e = nb.energy
  if e['nrows'] < nb.TB.bucketcount:
    return 1000
  if np.isnan(e['ra']):
    return 1000
  return int(-100 * math.log10(e['ra']))

def firstpass(remain, width=8, verbose=None):
  while len(remain.data) > 0:
    sn = NB.SubNet(width, remain.first_row().clientIP)
    rowmask = remain.data.clientIP.apply(sn.match)
    blk = remain.fork_block(sn, rowmask=rowmask)
    if verbose:
      print "Found:", blk.subnet.str(), \
             len(rowmask), len(blk.data), blk.energy['ra'], blk.rank
    NB.hpush(NB.NetBlock.todo, blk)
    remain.data = remain.data[~(rowmask)]
  return NB.NetBlock.todo

def parse_args():
  import argparse
  parser = argparse.ArgumentParser(
      description="Analyze a bunch of data for diurnal signals")
  parser.add_argument("--input", metavar="FILE", type=str,
                      default="../data.csv",
                      help="The file to process")
  parser.add_argument("--size", metavar="SIZE", type=int,
                      default=12,
                      help="Buckets per day")
  parser.add_argument("--width", metavar="SIZE", type=int,
                      default=8,
                      help="Buckets per day")
  parser.add_argument("--verbose", action='store_true',
                      help="verbose flag")
  return parser.parse_args()


def main():
  args = parse_args()
  verbose = args.verbose
  alldata = NB.NetBlock(NB.OneDay/args.size,
                   parse_row,
                   energy_mean_nan,
                   rank)
  alldata.parse(pd.read_csv(open(args.input)),
                downsample = 1000,
                cols = ALLCOLS)
  todo = firstpass(alldata, width = args.width, verbose=verbose)
  print len(todo), "Total Blocks"
  while len(todo):
    blk = NB.hpop(todo)
    print blk.subnet.str(), len(blk.data), blk.energy, blk.rank
  exit(0)

if __name__ == "__main__":
  main()
