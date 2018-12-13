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


# Label left intervals with -1, right with 1, and rest with 0
for trial in trials:
	conditions = [(df['\t'] >= trial[0]) & (df['\t'] <= trial[1]), # left
	(df['\t'] >= trial[2]) & (df['\t'] <= trial[3])] # right
	choices = [-1, 1]
	df['class'] = np.select(conditions, choices, default=0)

# MAKE A LIST OF INTERVALS
# Create a list of all the left light trials, next I will include the other 2 recordings above
lefts = [df.loc[(df['\t'] >= trials[0][0]) & (df['\t'] <= trials[0][1])]]
for trial in trials[1:]:
	lefts.append(df.loc[(df['\t'] >= trial[0]) & (df['\t'] <= trial[1])])

left_df = pd.DataFrame()
for left_interval in lefts:
	length = len(left_interval)
	left_df = left_df.append(left_interval.iloc[[length / 2]])
print(left_df)

# plt.plot(trial1_left['\t'], trial1_left['\t[\'Fp1\''])
# plt.show()

# Train an svm
df.drop(['Channel Names'], 1, inplace=True) # drop first
#print(df.head())
exit(0)
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, yTrain, yTest = model_selection.train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, yTrain)

accuracy = clf.score(X_test, yTest)
print(accuracy)

# example_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]])
# example_measures = example_measures.reshape(len(example_measures), -1)
# prediction = clf.predict(example_measures)
print(prediction)