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
from pandas import DataFrame, Series
import pandas as pd
import netblock as NB
import csv

# ResultTable = ScoreFrame(...) # one row per block, columns by properties

def grock(row, tb):
  row[tb.bucket(row["start_time"])] = row["download_mbps"]
  return row

def main():
  # NB: division between ScoreFrame() and main() remains TBD

  size = 12
  f = open("../data.csv", 'r')
  nb = NB.NetBlock(NB.OneDay/size)
  nb.parse(pd.read_csv(f), grock)
  print nb.data

  exit
  # split on extra dimensions (weeks, ISP, etc)
  # split by initial netblock size(e.g. /10)

  # score all blocks
  # insert into rank queue

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

def score(nb):
  de, te = nb.energy()
  return int(-100 * math.log(de / te))


if __name__ == "__main__":
  main()
