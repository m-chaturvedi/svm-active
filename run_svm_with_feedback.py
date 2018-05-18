from common import *
import svm
import svmutil
import display_image
import sklearn.metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class RunSVMWithFeedback:

  def __init__(self, random_feedback=False):
    self.X_train = []
    self.y_train = []
    self.labels_train = []

    self.X_test = None
    self.y_test = None
    self.labels_test = None
    self.svm_model = None

    self.all_distances = []
    self.num_refinements = CONFIG["svm"]["num_refinements"]
    self.num_for_feedback = CONFIG["svm"]["num_feedback"]
    self.display_image = None
    self.svm_param = CONFIG["svm"]["param"]
    self.simulated = CONFIG["svm"]["simulated"]
    self.random = random_feedback
    self.refinement_results = []

    logging.debug("LOADING DATA")
    self.all_data = self.load_all_data()
    logging.debug("LOADED DATA")

  def load_all_data(self):
    [all_pos_data, all_neg_data, all_pos_labels, all_neg_labels] = \
      np.load(CONFIG["database_config"]["all_data_file"])

    self.X_train, self.y_train, self.labels_train = [], [], []
    assert len(all_pos_data) + len(all_neg_data) == \
           len(all_pos_labels) + len(all_neg_labels)

    random_perm_pos = \
        np.random.permutation(len(all_pos_data))[:CONFIG["svm"]["num_pos"]]

    random_perm_neg = \
        np.random.permutation(len(all_neg_data))[:CONFIG["svm"]["num_neg"]]

    random_perm = np.append(random_perm_pos, len(all_pos_data) + random_perm_neg)
    np.random.shuffle(random_perm)

    all_data_list = all_pos_data + all_neg_data
    all_y_list = [1]*len(all_pos_data) + [-1]*len(all_neg_data)
    all_labels_list = all_pos_labels + all_neg_labels
    X_test = [all_data_list[i] for i in random_perm]
    self.y_test = [all_y_list[i] for i in random_perm]
    self.labels_test = [all_labels_list[i] for i in random_perm]

    assert sum(self.y_test) == \
           CONFIG["svm"]["num_pos"] - CONFIG["svm"]["num_neg"]

    self.X_test = [
      dict(zip(range(len(x)), x)) for x in X_test
    ]

  def calc_distance(self, x):
    svm_node_array, _ = svm.gen_svm_nodearray(x)
    distance = svmutil.svm_distance_from_plane(svm_node_array, self.svm_model)
    return abs(distance[0])

  def get_ind_for_feedback(self, X_test, use_random_indices):
    if use_random_indices:
      return np.random.choice(range(len(X_test)), self.num_for_feedback,
                                 replace=False)
    all_distances = []
    for x in X_test:
      all_distances.append(self.calc_distance(x))

    sorted_ind = np.argsort(all_distances)
    return sorted_ind[:self.num_for_feedback]

  def get_result_from_image(self, ind):
    if(self.simulated):
      return self._running_y_test[ind]

    label = self._running_labels_test[ind]
    display_img = display_image.DisplayPhotos()

    display_img.image_path = get_file_path_from_label(label) + "." + \
        CONFIG["database_config"]["image_extension"]

    logging.debug("DISPLAYING IMAGE " + display_img.image_path)
    display_img.display_window()
    return 1 if display_img.result[1] else -1

  def run_feedback(self, sorted_indices):
    for i in sorted_indices:
      res = self.get_result_from_image(i)
      self._running_X_train.append(self._running_X_test[i])
      self._running_y_train.append(res)
      self._running_labels_train.append(self._running_labels_test[i])

    for i in sorted(sorted_indices, reverse=True):
      self._running_X_test.pop(i)
      self._running_y_test.pop(i)
      self._running_labels_test.pop(i)

  def train_test_svm(self):
    logging.debug("TRAINING Samples: " + str(len(self._running_X_train)))
    logging.debug("TESTING Samples: " + str(len(self._running_X_test)))

    svm_problem = svm.svm_problem(self._running_y_train, self._running_X_train)
    self.svm_model = svmutil.svm_train(svm_problem, self.svm_param)
    predicted_labels, predicted_mse, predicted_probs = \
      svmutil.svm_predict(self._running_y_test,
                          self._running_X_test, self.svm_model, "-b 1")

    res = sklearn.metrics.accuracy_score(self._running_y_test, predicted_labels)
    self.refinement_results.append(res)
    print("RESULT: " + str(res*100))

  def run_svm(self):
    self._running_X_train = self.X_train[:]
    self._running_y_train = self.y_train[:]
    self._running_labels_train = self.labels_train[:]

    self._running_X_test = self.X_test[:]
    self._running_y_test = self.y_test[:]
    self._running_labels_test = self.labels_test[:]

    for i in range(self.num_refinements):
      sorted_ind = self.get_ind_for_feedback(self._running_X_test,
                                             True if i == 0 else self.random)
      self.run_feedback(sorted_ind)
      logging.debug("REFINEMENT ROUND: " + str(i+1))
      self.train_test_svm()


def main():
  svm_feedback_random = RunSVMWithFeedback(random_feedback=True)
  svm_feedback_random.run_svm()

  svm_feedback_non_random = RunSVMWithFeedback(random_feedback=False)
  svm_feedback_non_random.run_svm()

  p1 = [x*100 for x in svm_feedback_random.refinement_results]
  p2 = [x*100 for x in svm_feedback_non_random.refinement_results]

  plt.plot(p1, linewidth=2.0, label="random")
  plt.scatter(range(len(p1)), p1)

  plt.plot(p2, linewidth=2.0, label="non random")
  plt.scatter(range(len(p2)), p2)
  plt.xlabel("Refinements")
  plt.ylabel("Accuracy")
  plt.title("{} feedbacks per refinement and {} refinements".
            format(svm_feedback_random.num_for_feedback,
                   svm_feedback_random.num_refinements))
  plt.legend(loc="lower right")
  plt.ylim(0, 100)
  plt.xlim(0, CONFIG["svm"]["num_refinements"])
  plt.savefig(CONFIG["svm"]["plot_file"])


if __name__ == "__main__":
  main()


