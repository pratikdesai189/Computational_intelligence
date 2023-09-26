import time
import os

from matplotlib.pyplot import subplot
from numpy import ones
from numpy.linalg import norm

from dtw import dtw
import librosa.display
from sklearn.neighbors import KNeighborsClassifier

y1, sr1 = librosa.load('../../train/fcmc0-a1-t.wav')
y2, sr2 = librosa.load('../../train/fcmc0-b1-t.wav')

subplot(1, 2, 1)
mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1)
librosa.display.specshow(mfcc1)

subplot(1, 2, 2)
mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2)
librosa.display.specshow(mfcc2)

# Calculate the DTW between the 2 sample audios 'a' and 'b'
dist, cost, path, _ = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
print('Normalized distance between the two sounds:', dist)

dirname = "../../train"
files = [f for f in os.listdir(dirname) if not f.startswith('.')]

# The following code Iterates through the Training folder and builds the trained representation in the Distance matrix
start = time.perf_counter()
minval = 200
distances = ones((len(files), len(files)))
y = ones(len(files))

for i in range(len(files)):
    y1, sr1 = librosa.load(dirname + "/" + files[i])
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1)
    for j in range(len(files)):
        y2, sr2 = librosa.load(dirname + "/" + files[j])
        mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2)
        dist, _, _, _ = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))

        distances[i, j] = dist
    if i % 2 == 0:
        y[i] = 0  # 'a'
    else:
        y[i] = 1  # 'b'
print("Time used: {}s".format(time.perf_counter() - start))

label = ['a', 'b']
# Train KNN Classifier
classifier = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
classifier.fit(distances, y)

y, sr = librosa.load('../../test/farw0-b1-t.wav')
mfcc = librosa.feature.mfcc(y=y, sr=sr)
distanceTest = []
for i in range(len(files)):
    y1, sr1 = librosa.load(dirname + "/" + files[i])
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1)
    dist, _, _, _ = dtw(mfcc.T, mfcc1.T, dist=lambda x, y: norm(x - y, ord=1))
    distanceTest.append(dist)

pre = classifier.predict([distanceTest])[0]
print(pre)

result = label[int(pre)]

print("Predict audio is: '{}'".format(result))
