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

def parse_row(nb, row, tb):
  row["clientIP"] = NB.inet_addr(row["client_ip_v4"])
  row[tb.bucket(row["start_time"])] = row["download_mbps"]
  return row

def energy_mean_nan(nb, harmonics=2, method='ffill'):
  """Compute the energy of this NetBlock.

  Returns a dictionary:
  e24 - energy at 1/(24hr) and selected harmonics
  te - total energy
  """

  timeseries = Series([nb.data[tt].mean() for tt in  nb.TBall])
  nan = nb.TB.bucketcount - timeseries.count()
  if method:
    timeseries = timeseries.fillna(method=method)
    # TODO (mattmathis) exlore limit=
  else:
    timeseries = [0.0 if np.isnan(tv) else tv for tv in timeseries]
  e24, te = energy.power_ratio(timeseries, len(timeseries), harmonics)
  try:
    ra = e24/te
  except:
    # failed to fill all NaNs
    ra = float('nan')
  return { 'e24':e24, 'te':te, 'ra':ra, "nrows":len(nb.data), 'nan':nan }

def rank(nb):
  e = nb.energy
  if e['nrows'] < nb.TB.bucketcount:
    return 1000
  if np.isnan(e['ra']):
    return 1000
  return int(-100 * math.log10(e['ra']))

def main():
  # NB: division between ScoreFrame() and main() remains TBD

  verbose = False
  size = 12
  Swidth = 8
  file = "../data.csv"
  if (len(sys.argv) >= 2):
    file = sys.argv[1]
  f = open(file, 'r')
  alldata = NB.NetBlock(NB.OneDay/size,
                   parse_row,
                   energy_mean_nan,
                   rank)
  alldata.parse(pd.read_csv(f),
                downsample = 1000,
                cols=['server_ip_v4', 'client_ip_v4',
                      'start_time', 'Duration',
                      'download_mbps', 'min_rtt', 'avg_rtt',
                      'retran_per_DataSegsOut', 'retran_per_CongSignals'])
  remain = alldata
  while len(remain.data) > 0:
    sn = NB.SubNet(Swidth, remain.first_row().clientIP)
    rowmask = remain.data.clientIP.apply(sn.match)
    blk = remain.fork_block(sn, rowmask=rowmask)
    if verbose:
      print "Found:", blk.subnet.str(), \
             len(rowmask), len(blk.data), blk.energy['ra'], blk.rank
    NB.hpush(NB.NetBlock.todo, blk)
    remain.data = remain.data[~(rowmask)]

  print len(NB.NetBlock.todo), "Total Blocks"
  while len(NB.NetBlock.todo):
    blk = NB.hpop(NB.NetBlock.todo)
    print blk.subnet.str(), len(blk.data), blk.energy, blk.rank
  exit(0)

  # while not done:
    # pop highest rank block
    # split it on one bit
    # score each half
    # if scores are similar:
      # put parent on done list
     #  discard children
    # else:
      # insert children into rank queue by score

    # sort done queue by rank
    # display top ranked netblocks


if __name__ == "__main__":
  main()
