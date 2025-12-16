import numpy as np
import pyedflib

file_name = pyedflib.data.get_generator_filename()
f = pyedflib.EdfReader("/home/tcosse/total-perspective-vortex/data/physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf")
n = f.signals_in_file
signal_labels = f.getSignalLabels()
print (signal_labels)
sigbufs = np.zeros((n, f.getNSamples()[0]))
for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)
print(sigbufs)
annotation = np.zeros((n, f.read_annotation()[0]))
for i in np.arange(n):
    annotation[i, :] = f.read_annotation()
print(annotation)