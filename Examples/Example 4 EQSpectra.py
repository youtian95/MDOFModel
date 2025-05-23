from pathlib import Path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import eqsig.single

parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

bf, sub_fig = plt.subplots()
with open(os.path.join(os.path.dirname(__file__),'H-E12140.dat'), "r") as f:
    Allstr = f.read()
Allstr = Allstr.split()
a = np.array(Allstr).astype(float)
print(a.max())
dt = 0.005  # time step of acceleration time series
periods = np.linspace(0, 5, 100)  # compute the response for 100 periods between T=0.2s and 5.0s
record = eqsig.AccSignal(a * 9.8, dt)
record.generate_response_spectrum(response_times=periods)
times = record.response_times

sub_fig.plot(times, record.s_a/9.8, label="eqsig")
plt.show()