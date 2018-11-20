import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm

with open('SubjectMichelle___timestamp_1501862371.27/SubjectMichelle__log.txt') as f:
	lines = [line.rstrip('\n').replace('\'', '"') for line in f]
trials = []
# Get start and end times for left and right lights and rest intervals for each trial
for line in lines[1:]:
	times = json.loads(line)
	trials.append([times['start_left_time'], times['end_left_time'], times['start_right_time'], times['end_right_time'], times['start_rest1_time'], times['end_rest1_time'], times['start_rest2_time'], times['end_rest2_time']])

df = pd.read_csv('SubjectMichelle___timestamp_1501862371.27/SubjectMichelle__eeg.csv', header=8)
# df2 = pd.read_csv('SubjectMichelle___timestamp_1501862954.25/SubjectMichelle__eeg.csv', header=8)
# df3 = pd.read_csv('SubjectMichelle___timestamp_1501863670.92/SubjectMichelle__eeg.csv', header=8)
# Create a list of all the left light trials, next I will include the other 2 recordings above
lefts = [df.loc[(df['\t'] >= trials[0][0]) & (df['\t'] <= trials[0][1])]]
for trial in trials[1:]:
	lefts.append(df.loc[(df['\t'] >= trial[0]) & (df['\t'] <= trial[1])])
trial1_left = lefts[0]
plt.plot(trial1_left['\t'], trial1_left['\t[\'Fp1\''])
plt.show()