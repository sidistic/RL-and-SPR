import utilities as utils
from scipy.io import wavfile
import numpy as np

# Read the files as numpy array
rate1, data1 = wavfile.read("sourceA.wav")
rate2, data2 = wavfile.read("sourceB.wav")

# Using the mixSounds helper function from utilities.py
mixedX = utils.mixSounds([data1, data2], [0.3, 0.7]).astype(np.int16)
mixedY = utils.mixSounds([data1, data2], [0.6, 0.4]).astype(np.int16)

# Plot the mixed sound sources
utils.plotSounds([mixedX, mixedY], ["mixedA","mixedB"], rate1, "../plots/sounds/Ring_StarWars_mixed", False)

# Save the mixed sources as wav files
wavfile.write("mixedA.wav", rate1, mixedX)
wavfile.write("mixedB.wav", rate1, mixedY)