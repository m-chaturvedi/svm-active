import sys
sys.path.append("/home/chaturvedi/workspace/video-cat/build_gcc5/install/lib/python2.7/dist-packages")
import svm
import svmutil
import pdb
import pickle
import logging
import numpy
import display_image
import config
import sklearn.metrics
import matplotlib.pyplot as plt

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class RunSVMWithFeedback:

  def __init__(self, random_feedback=False, simulated=True):
    logging.debug("LOADING MODEL")
    self.svm_model = svmutil.svm_load_model("svm_model.txt")
    logging.debug("LOADING PICKLE DATA")
    self.all_data = pickle.load(open("svm_model_data.pickle", "rb"))
    logging.debug("LOADED PICKLE DATA")

    self.X_test = self.all_data["X_test"]
    self.y_test = self.all_data["y_test"]
    self.labels_test = self.all_data["labels_test"]

    self.X_train = self.all_data["X_train"]
    self.y_train = self.all_data["y_train"]
    self.labels_train = self.all_data["labels_train"]

    self.predicted_probs = self.all_data["predicted_prob"]
    self.all_distances = []
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    self.num_refinements = 15
    self.num_for_feedback = 20
    self.keras_config = config.TryKerasConfig(getting_features=False)
    self.display_image = None
    self.svm_param = svm.svm_parameter("-b 1 -t 5 -c 2 -g 0.0078125")
    self.simulated = simulated
    self.random = random_feedback
    self.refinement_results = []

  def calc_distance(self, x):
    svm_node_array, _ = svm.gen_svm_nodearray(x)
    distance = svmutil.svm_distance_from_plane(svm_node_array, self.svm_model)
    return abs(distance[0])

  def get_ind_for_feedback(self, X_test):
    if self.random:
      return numpy.random.choice(range(len(X_test)), self.num_for_feedback,
                                 replace=False)
    all_distances = []
    for x in X_test:
      all_distances.append(self.calc_distance(x))

    sorted_ind = numpy.argsort(all_distances)
    return sorted_ind[:self.num_for_feedback]

  def get_result_from_image(self, ind):
    if(self.simulated):
      return self._running_y_test[ind]

    label = self._running_labels_test[ind]
    display_img = display_image.DisplayPhotos()
    display_img.image_path = \
      self.keras_config.get_file_path_from_label(
        label) + self.keras_config.img_extension
    logging.debug("DISPLAYING IMAGE " + display_img.image_path)
    display_img.display_window()
    return display_img.result[1]

  def run_feedback(self, sorted_indices):
    feedback_results = []
    for i in sorted_indices:
      res = self.get_result_from_image(i)
      feedback_results.append(1 if res else -1)

      self._running_X_train.append(self._running_X_test[i])
      self._running_y_train.append(self._running_y_test[i])
      self._running_labels_train.append(self._running_labels_test[i])

    for i in sorted(sorted_indices, reverse=True):
      self._running_X_test.pop(i)
      self._running_y_test.pop(i)
      self._running_labels_test.pop(i)

  def train_test_svm(self):
    X_train_dict = [dict(zip(range(len(x)), x)) for x in self._running_X_train]
    X_test_dict = [dict(zip(range(len(x)), x)) for x in self._running_X_test]

    svm_problem = svm.svm_problem(self._running_y_train, X_train_dict)
    self.svm_model = svmutil.svm_train(svm_problem, self.svm_param)
    predicted_labels, predicted_mse, predicted_probs = \
      svmutil.svm_predict(self._running_y_test,
                          X_test_dict, self.svm_model, "-b 1")

    res = sklearn.metrics.accuracy_score(self._running_y_test, predicted_labels)
    self.refinement_results.append(res)
    print("RESULT: " + str(res*100))


  def run_svm(self):
    self._running_X_train = self.X_train[:]
    self._running_X_test = self.X_test[:]
    self._running_labels_train = self.labels_train[:]

    self._running_y_train = self.y_train[:]
    self._running_y_test = self.y_test[:]
    self._running_labels_test = self.labels_test[:]


    for i in range(self.num_refinements):
      sorted_ind = self.get_ind_for_feedback(self._running_X_test)
      self.run_feedback(sorted_ind)
      self.train_test_svm()


def main():
  svm_feedback_random = RunSVMWithFeedback(random_feedback=True)
  svm_feedback_random.run_svm()

  svm_feedback_non_random = RunSVMWithFeedback(random_feedback=False)
  svm_feedback_non_random.run_svm()

  p1 = [x*100 for x in svm_feedback_random.refinement_results]
  p2 = [x*100 for x in svm_feedback_non_random.refinement_results]

  plt.plot(p1, linewidth=2.0)
  plt.scatter(range(len(p1)), p1)

  plt.plot(p2, linewidth=2.0)
  plt.scatter(range(len(p2)), p2)
  plt.xlabel("Refinements")
  plt.ylabel("Accuracy")
  plt.title("Accuracies for {} feedbacks and {} refinements".
            format(svm_feedback_random.num_for_feedback,
                   svm_feedback_random.num_refinements))
  plt.savefig("refinements.png")


if __name__ == "__main__":
  main()


