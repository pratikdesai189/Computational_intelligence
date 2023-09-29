import argparse
import time
import os

from numpy import ones
from numpy.linalg import norm

from dtw import dtw
import librosa.display
from sklearn.neighbors import KNeighborsClassifier


def train_predict_voice_command(test_command):
    global mfcc
    y1, sr1 = librosa.load('../../train/fcmc0-a1-t.wav')
    y2, sr2 = librosa.load('../../train/fcmc0-b1-t.wav')
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2)
    # Calculate the DTW between the 2 sample audios 'a' and 'b'
    dist, cost, path, _ = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
    print('Normalized distance between the two sounds:', dist)
    dirname = "../../train"
    files = [f for f in os.listdir(dirname) if not f.startswith('.')]

    # The following code Iterates through the Training folder
    # and builds the trained representation in the Distance matrix
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
    test_command = "../../test/" + test_command
    y, sr = librosa.load(test_command)
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
    print("Predicted audio is: '{}'".format(result))


def get_arguments():
    parser = argparse.ArgumentParser(description='Voice_Command_Detection_Using_DTW')
    parser.add_argument('test_command', type=str, help='test_command_a_or_b')
    result = parser.parse_args()
    return result


def main():
    args = get_arguments()
    train_predict_voice_command(args.test_command)


if __name__ == '__main__':
    main()
