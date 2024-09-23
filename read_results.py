import sys
import numpy as np

tasks = {}
with open(sys.argv[1], 'r') as f:
    for l in f.readlines():
        if l.startswith('Reached target accuracy of'):
            pos = l.find('>=')
            task = l[pos + 2: pos + 6]
            if task not in tasks:
                tasks[task] = []
            tasks[task].append(float(l.split(' ')[6]))

for task in tasks:
    if len(tasks[task]) < 10:
        print(task, len(tasks[task]))
        tasks[task].extend([20000] * (10 - len(tasks[task])))
    print(task, len(tasks[task]), np.median(tasks[task]))