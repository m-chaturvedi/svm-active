from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pdb
import sys
sys.path.append("/home/chaturvedi/workspace/video-cat/build_gcc5/install/lib/python2.7/dist-packages")
import svm
import pickle
import svmutil
import logging
from sklearn.model_selection import train_test_split
import pickle
import numpy

class TrySVMWithKeras:

    def __init__(self):
        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

        self.labels_train = None
        self.labels_test = None

        self.pickle_file = "loaded_image_vectors.pickle"

        self.num_pos = 250
        self.pos_neg_ratio = 1
        self.test_train_split = 1 - 0.025

        self.svm_problem = None
        # Default kernel type is radial basis
        # Default cost is 1
        # Using exactly as we use use in our SVM
        self.svm_param = svm.svm_parameter("-b 1 -t 5 -c 2 -g 0.0078125")
        self.svm_model = None
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    def run_svm(self, save_to_pickle):
        logging.debug("LOADING FROM PICKLE")
        self.load_from_pickle()
        self.svm_problem = svm.svm_problem(self.y_train, self.X_train)
        logging.debug("TRAINING SVM")
        self.svm_model = svmutil.svm_train(self.svm_problem, self.svm_param)

        predicted_labels, predicted_mse, predicted_probs = \
            svmutil.svm_predict(self.y_test,
                                self.X_test, self.svm_model, "-b 1")

        if save_to_pickle:
            svmutil.svm_save_model("svm_model.txt", self.svm_model)
            data_to_dump = {
                "X_train": self.X_train,
                "y_train": self.y_train,
                "y_test": self.y_test,
                "X_test": [[x[i] for i in range(len(x))] for x in self.X_test],
                "predicted_prob": predicted_probs,
                "labels_train": self.labels_train,
                "labels_test": self.labels_test
            }
            logging.debug("SAVING TO PICKLE")
            pickle.dump(data_to_dump, open("svm_model_data.pickle", "wb"))

        return predicted_probs

    def load_from_pickle(self):
        (all_pos_data, all_neg_data, all_pos_labels, all_neg_labels) = \
            pickle.load(open(self.pickle_file, "rb"))

        num_neg = int(self.num_pos / self.pos_neg_ratio)

        X_train, X_test, self.y_train, self.y_test, \
        self.labels_train, self.labels_test = \
            train_test_split(
                all_pos_data[:self.num_pos] + all_neg_data[:num_neg],
                [1] * self.num_pos + [-1] * num_neg,
                all_pos_labels[:self.num_pos] + all_neg_labels[:num_neg],
                test_size=self.test_train_split, #random_state=42
            )

        self.X_train = [
            dict(zip(range(len(x)), x)) for x in X_train
        ]

        self.X_test = [
            dict(zip(range(len(x)), x)) for x in X_test
        ]


if __name__ == "__main__":
    try_svm = TrySVMWithKeras()
    try_svm.run_svm(True)

