import os
import pdb
import pandas
import numpy as np
import re
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import pickle

class TryKerasConfig:

    def __init__(self, getting_features=True):
        self.label_for_training = "broom"
        self.image_loader = "KERAS"

        self.base_directory_name = "tiny-imagenet-200"
        self.image_names_file = "my_words.txt"
        self.image_base_dir_rel_path = "train"
        self.ASSUMED_NUMBER_OF_IMAGES_IN_CATEGORY = 500
        self.pickle_file = "loaded_image_vectors.pickle"

        if getting_features:
            self.model = VGG16(weights='imagenet', include_top=False)
        self.target_size = (224, 224)
        self.img_extension = ".JPEG"


    def load_image_names(self):
        image_names_full_path = os.path.join(self.base_directory_name,
                                             self.image_names_file)
        # This tends to interpret nan as a float
        self.image_data = pandas.read_csv(image_names_full_path, sep="\t",
                                          header=None).as_matrix()
        self.names = [str(i) for i in self.image_data[:, 1]]
        self.ids = [str(i) for i in self.image_data[:, 0]]

    def load_image_to_vector(self, img_path):
        if self.image_loader == "KERAS":
            img = image.load_img(img_path, target_size=self.target_size)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = self.model.predict(x)

        return features.flatten()

    def get_file_path_from_label(self, label):
        regex_match = re.match('(.*)_(.*)', label)
        assert len(regex_match.groups()) == 2
        assert regex_match.groups()[0][0] == 'n'
        assert regex_match.groups()[1].isdigit()
        class_dir_name = regex_match.groups()[0]
        file_path = os.path.join(*[self.base_directory_name,
                                   self.image_base_dir_rel_path,
                                   class_dir_name, "images", label])

        return file_path

    def load_image_data_from_labels(self, labels):
        base_path = os.path.join(self.base_directory_name,
                                 self.image_base_dir_rel_path)
        all_vectors = []
        for label in labels:
            file_path = self.get_file_path_from_label(label)
            vec = self.load_image_to_vector(file_path + self.img_extension)
            all_vectors.append(vec)

        return all_vectors

    def save_all_data(self, training_label):
        self.load_image_names()
        filtered_names = \
            [x.upper() == training_label.upper() for x in self.names]
        assert sum(filtered_names) == 1, "ONE LABEL EXPECTED"
        label_ind = self.ids[self.names.index(training_label)]

        all_pos_labels = []
        all_neg_labels = []

        image_range = range(self.ASSUMED_NUMBER_OF_IMAGES_IN_CATEGORY)
        for i in image_range:
            all_pos_labels.append(label_ind+ "_" + str(i))
            for id in self.ids:
                if id is label_ind:
                    continue
                all_neg_labels.append(id + "_" + str(i))

        all_pos_data = self.load_image_data_from_labels(all_pos_labels)
        all_neg_data = self.load_image_data_from_labels(all_neg_labels)
        data_to_dump = (all_pos_data, all_neg_data, all_pos_labels, all_neg_labels)

        pickle.dump(data_to_dump, open(self.pickle_file, "wb"))

        return (all_pos_data, all_neg_data)


if __name__ == "__main__":
    try_keras = TryKerasConfig()
    try_keras.save_all_data("broom")

