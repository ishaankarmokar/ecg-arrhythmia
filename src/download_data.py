"""Download the 44 non-paced MIT-BIH Arrhythmia Database records from PhysioNet.

Owner: Ishaan Karmokar

Only the records named in config.DS1 + config.DS2 are fetched (signal .dat,
header .hea and reference annotations .atr), about 80 MB in total. Records
already on disk are skipped, so this is safe to re-run.

    python src/download_data.py
"""
import os
import sys

import wfdb

sys.path.insert(0, os.path.dirname(__file__))
from config import DS1, DS2

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'data', 'raw')


def main():
    os.makedirs(RAW, exist_ok=True)
    wanted = DS1 + DS2
    missing = [r for r in wanted
               if not all(os.path.exists(os.path.join(RAW, f'{r}.{ext}'))
                          for ext in ('dat', 'hea', 'atr'))]
    if not missing:
        print(f'all {len(wanted)} records already in {RAW}')
        return
    print(f'downloading {len(missing)} records from PhysioNet (mitdb) ...')
    wfdb.dl_database('mitdb', RAW, records=missing, annotators=['atr'])
    print('done')


if __name__ == '__main__':
    main()
