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
"""Thin a dataset

thin.py <file> <ratio>

e.g. "thin.py foo.csv 100" creates foo_thin100.csv with 1/100 the rows.

"""

import sys
import pandas as pd
import numpy as np
import random

def main():
  file = sys.argv[1]
  ofile = file
  assert(ofile[-4:] == ".csv")
  ofile = ofile[:-4]+"_thin"+sys.argv[2]+".csv"
  print file, ofile

  df = pd.read_csv(file)
  #  df = df.sample(sys.argv[2]) We don't have sample?
  mask=pd.Series(np.random.randint(0, int(sys.argv[2]), len(df)) == 0)
  df[mask].to_csv(ofile)

if __name__ == "__main__":
  main()
