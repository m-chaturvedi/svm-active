from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pdb
import svm
import pickle
import svmutil
import logging
from sklearn.model_selection import train_test_split
import numpy

# model = VGG16(weights='imagenet', include_top=False)
#
# img_path = 'n01774384_1.JPEG'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#
# features = model.predict(x)


class TrySVMWithKeras:

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.pickle_file = "loaded_image_vectors.pickle"

        self.num_pos = 500
        self.pos_neg_ratio = 1
        self.test_train_split = 0.5

        self.svm_problem = None
        # Default kernel type is radial basis
        # Default cost is 1
        self.svm_param = svm.svm_parameter("-b 1")
        self.svm_model = None
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    # Copy-paste from here:
    # https://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python
    def PCA(self, data, dims_rescaled_data=2):
        """
        returns: data transformed in 2 dims/columns + regenerated original data
        pass in: data as 2D NumPy array
        """
        import numpy as NP
        from scipy import linalg as LA
        m, n = data.shape
        # mean center the data
        data -= data.mean(axis=0)
        # calculate the covariance matrix
        R = NP.cov(data, rowvar=False)
        # calculate eigenvectors & eigenvalues of the covariance matrix
        # use 'eigh' rather than 'eig' since R is symmetric,
        # the performance gain is substantial
        evals, evecs = LA.eigh(R)
        # sort eigenvalue in decreasing order
        idx = NP.argsort(evals)[::-1]
        evecs = evecs[:,idx]
        # sort eigenvectors according to same index
        evals = evals[idx]
        # select the first n eigenvectors (n is desired dimension
        # of rescaled data array, or dims_rescaled_data)
        evecs = evecs[:, :dims_rescaled_data]
        # carry out the transformation on the data using eigenvectors
        # and return the re-scaled data, eigenvalues, and eigenvectors
        return NP.dot(evecs.T, data.T).T, evals, evecs

    def run_svm(self):
        logging.debug("LOADING FROM PICKLE")
        self.load_from_pickle()
        self.svm_problem = svm.svm_problem(self.y_train, self.X_train)
        logging.debug("TRAINING SVM")
        self.svm_model = svmutil.svm_train(self.svm_problem, self.svm_param)

        predicted_labels, predicted_mse, predicted_probs = \
            svmutil.svm_predict(self.y_test,
                                self.X_test, self.svm_model, "-b 1")

        pdb.set_trace()

    def load_from_pickle(self):
        (all_pos_data, all_neg_data) = \
            pickle.load(open(self.pickle_file, "rb"))

        num_neg = int(self.num_pos / self.pos_neg_ratio)

        X_train, X_test, self.y_train, self.y_test = \
            train_test_split(
                all_pos_data[:self.num_pos] + all_neg_data[:num_neg],
                [1] * self.num_pos + [-1] * num_neg,
                test_size=self.test_train_split, random_state=42
            )

        pdb.set_trace()
        # X_train = list(self.PCA(numpy.array(X_train)))
        # X_test = list(self.PCA(numpy.array(X_test)))
        self.X_train = [
            dict(zip(range(len(x)), x)) for x in X_train
        ]

        self.X_test = [
            dict(zip(range(len(x)), x)) for x in X_test
        ]


if __name__ == "__main__":
    try_svm = TrySVMWithKeras()
    try_svm.run_svm()

