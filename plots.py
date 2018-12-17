import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm
from scipy.signal import welch

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
# Classify all the 
# for index, row in df.iterrows():
# 	one_sec = df.O1[:500]
df.drop(['Channel Names'], 1, inplace=True) # drop first
# Label left intervals with -1, right with 1, and rest with 0
# df['class'] = [0] * len(df.index)
# for trial in trials:
# 	conditions = [(df['\t'] >= trial[0]) & (df['\t'] <= trial[1]), # left
# 	(df['\t'] >= trial[2]) & (df['\t'] <= trial[3])] # right
# 	choices = [-1, 1]
# 	df['class'] = np.select(conditions, choices, df['class'])

# make a list of intervals
lefts = []
rights = []
rest1 = []
rest2 = []
for trial in trials[1:]:
	lefts.append(df.loc[(df['\t'] >= trial[0]) & (df['\t'] <= trial[1])])
	rights.append(df.loc[(df['\t'] >= trial[2]) & (df['\t'] <= trial[3])])
	rest1.append(df.loc[(df['\t'] >= trial[4]) & (df['\t'] <= trial[5])])
	rest2.append(df.loc[(df['\t'] >= trial[6]) & (df['\t'] <= trial[7])])

def fourier(intervals, X, y, label):
	for interval in intervals:
		for i in range(0, len(interval.index) // 500):
			one_sec = interval.O1[i * 500 : (i+1)*500]
			X.append(np.concatenate(welch(one_sec, fs=500, nperseg=500)))
			y.append(label)

X = []
y = []
fourier(lefts, X, y, -1)
fourier(rights, X, y, 1)
fourier(rest1, X, y, 0)
fourier(rest2, X, y, 0)
X = pd.DataFrame(X)

# Train an svm
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print('Accuracy: ', accuracy)