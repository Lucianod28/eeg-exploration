import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm

with open('SubjectMichelle___timestamp_1501862371.27/SubjectMichelle__log.txt') as f:
	lines = [line.rstrip('\n').replace('\'', '"') for line in f]
trial_1 = json.loads(lines[1])
start_left = trial_1['start_left_time']
end_left = trial_1['end_left_time']
start_right = trial_1['start_right_time']
end_right = trial_1['end_right_time']

df = pd.read_csv('SubjectMichelle___timestamp_1501862371.27/SubjectMichelle__eeg.csv', header=8)
print(df.head())
left = df.loc[(df['\t'] >= start_left) & (df['\t'] <= end_left)]
print(left)

plt.plot(left['\t'], left['\t[\'Fp1\''])
plt.show()