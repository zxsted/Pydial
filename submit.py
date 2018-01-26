from __future__ import print_function

import os
import thread
import time
import sys

assert len(sys.argv) == 2

finished = 0


def submit(seed):
    os.system('mkdir ' + sys.argv[1] + '_log')
    runfile = open(sys.argv[1] + '_log/run' + str(seed) + '.sh', 'w')
    runfile.write('#!/bin/sh\n')
    runfile.write('/mnt/speechlab/users/bwt09/packages/anaconda2/bin/python pydial.py train '
                  'config/other_configs/' + sys.argv[1] + '.cfg --seed=' + str(seed) + '\n')
    while True:
        os.system('sbatch -p cpuq -n 2 -o ' + sys.argv[1] + '_log/LOG' + str(seed) + ' run' + str(seed) + '.sh')

        complete = False
        error = False
        while (complete or error) is False:
            while True:
                try:
                    result_file = open(sys.argv[1] + '_log/LOG' + str(seed), 'r')
                except:
                    continue
                finally:
                    break
            for u in result_file:
                if u.find('Training complete') != -1:
                    complete = True
                    break
                if u.find('I/O Error') != -1:
                    error = True
                    break
        if error:
            continue
        elif complete:
            break

    print(sys.argv[1], 'seed', str(seed), 'finished')
    print(finished, 'finished')


try:
    for seed in range(0, 10):
        thread.start_new_thread(submit, (seed, ))
except:
    print('Error!')

while True:
    pass
