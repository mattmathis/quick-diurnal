#!/bin/bash

# hardcoded prototype for Cogent NYC (lga02) 1 Jan 2014 - 1 Apr 2014

bq query --format=csv '
SELECT
  web100_log_entry.connection_spec.local_ip AS server_ip_v4,
  web100_log_entry.connection_spec.remote_ip AS client_ip_v4,
  8 * (web100_log_entry.snap.HCThruOctetsAcked /
        (web100_log_entry.snap.SndLimTimeRwin +
         web100_log_entry.snap.SndLimTimeCwnd +
         web100_log_entry.snap.SndLimTimeSnd)) AS download_mbps,
  (web100_log_entry.snap.SumRTT /
     web100_log_entry.snap.CountRTT) AS avg_rtt,
  web100_log_entry.snap.MinRTT AS min_rtt,
  (web100_log_entry.snap.SegsRetrans /
     web100_log_entry.snap.DataSegsOut) AS packet_retransmit_rate
FROM
  [plx.google:m_lab.ndt.all]
WHERE
  web100_log_entry.log_time IS NOT NULL
  AND web100_log_entry.snap.SndLimTimeCwnd IS NOT NULL
  AND web100_log_entry.snap.SndLimTimeRwin IS NOT NULL
  AND web100_log_entry.connection_spec.local_ip IN
	("38.106.70.147", "38.106.70.173", "38.106.70.160" )
  AND web100_log_entry.log_time > PARSE_UTC_USEC("2014-01-01 00:00:00") / POW(10, 6)
  AND web100_log_entry.log_time < PARSE_UTC_USEC("2014-04-01 00:00:00") / POW(10, 6)
  AND project = 0
  AND web100_log_entry.is_last_entry = True
  AND connection_spec.data_direction = 1
  AND web100_log_entry.snap.CongSignals > 0
  AND web100_log_entry.snap.HCThruOctetsAcked >= 8192
  AND (web100_log_entry.snap.State == 1
    OR (web100_log_entry.snap.State >= 5
        AND web100_log_entry.snap.State <= 11))
  AND (web100_log_entry.snap.SndLimTimeRwin +
       web100_log_entry.snap.SndLimTimeCwnd +
       web100_log_entry.snap.SndLimTimeSnd) >= 9000000
  AND (web100_log_entry.snap.SndLimTimeRwin +
       web100_log_entry.snap.SndLimTimeCwnd +
       web100_log_entry.snap.SndLimTimeSnd) < 3600000000
  AND web100_log_entry.snap.CountRTT > 10

'  > data.csv
