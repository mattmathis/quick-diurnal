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

Shard the data by high bits (default to 8)

We keep a computed summary scores for reference but strip added columns


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
  row["Value"] = row["download_mbps"]
  row[tb.bucket(row["start_time"])] = row["Value"]
#  row[tb.bucket(row["start_time"])] = row["avg_rtt"]
  return row

def rank(nb):
  """Provide a preliminary estimate of interest"""
  e = nb.energy
  if e['nrows'] < nb.TB.bucketcount:
    return 1000
  if np.isnan(e['ratio']):
    return 1000
  return (-100 * math.log10(e['ratio']))

# TODO move this into the netblock class
def firstpass(remain, width=8, verbose=None):
  while len(remain.data) > 0:
    sn = NB.SubNet(width, remain.first_row().clientIP)
    rowmask = remain.data.clientIP.apply(sn.match)
    blk = remain.fork_block(sn, rowmask=rowmask)
    if verbose:
      print "Found:", blk.subnet.str(), \
             len(rowmask), len(blk.data), blk.energy['ratio'], blk.rank
    NB.hpush(NB.NetBlock.todo, blk)
    remain.data = remain.data[~(rowmask)]
  print len(NB.NetBlock.todo)
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
  parser.add_argument("--downsample", metavar="COUNT", type=int,
                      default=1000,
                      help="Buckets per day")
  parser.add_argument("--width", metavar="SIZE", type=int,
                      default=8,
                      help="IP mask width")
  parser.add_argument("--verbose", action='store_true',
                      help="verbose flag")
  return parser.parse_args()


def main():
  args = parse_args()
  verbose = args.verbose
  basename = args.input
  if basename[-4:] != '.csv':
    print "Input file must be .csv"
    exit (2)
  basename = basename[:-4] + '.slice.'
  alldata = NB.NetBlock(NB.OneDay/args.size,
                   parse_row,
                   NB.NetBlock.norm_spectra,
                   rank=rank)
  alldata.parse(pd.read_csv(open(args.input)),
                downsample = args.downsample,
                cols = ALLCOLS)
  remain = alldata
  while len(remain.data) > 0:
    sn = NB.SubNet(args.width, remain.first_row().clientIP)
    rowmask = remain.data.clientIP.apply(sn.match)
    blk = remain.fork_block(sn, rowmask=rowmask)
    if verbose:
      print "Found:", blk.subnet.str(), \
             len(rowmask), len(blk.data), blk.energy['ratio'], blk.rank
    NB.hpush(NB.NetBlock.todo, blk)
    remain.data = remain.data[~(rowmask)]

  summary = []
  sumcols = ['subnet', 'nrows', 'nan', 'sum24', 'tsig', 'ratio', 'rank']
  while len(NB.NetBlock.todo):
    blk = NB.hpop(NB.NetBlock.todo)
    ofile = basename + blk.subnet.sstr() + '.csv'
    blk.data.to_csv(ofile, cols=ALLCOLS, index=False)
    e =  {"subnet" : blk.subnet.str(), "rank": blk.rank }
    e.update(blk.energy)
    summary.append(e)
  ofile = basename + 'summary.csv'
  df = DataFrame(summary)
  df = df.sort('nrows', ascending = False)
  df.to_csv(ofile, cols=sumcols, index=False)

if __name__ == "__main__":
  main()
