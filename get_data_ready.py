import pandas
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from subprocess import call
from common import *


class TryKerasConfig:

    def __init__(self):
        database_config = CONFIG["database_config"]
        self.label_for_training = database_config["label_for_training"]
        self.base_directory_name = database_config["base_directory_name"]
        self.image_names_file = database_config["image_names_file"]
        self.image_base_dir_rel_path = \
            database_config["image_base_dir_rel_path"]

        self.ASSUMED_NUMBER_OF_IMAGES_IN_CATEGORY = \
            database_config["ASSUMED_NUMBER_OF_IMAGES_IN_CATEGORY"]

        self.pickle_file = database_config["all_data_file"]

        self.model = VGG16(weights='imagenet', include_top=False)
        self.target_size = tuple(database_config["target_size"])
        self.img_extension = database_config["image_extension"]
        self.download_database()

    def download_database(self):
        if CONFIG["database_config"]["download_data"]:
            assert call(["wget", CONFIG["database_config"]["repo_url"]]) == 0, \
                 "ERROR DOWNLOADING FILE"
            logging.debug("DOWNLOADING AND UNZIPPING DATABASE")
            assert call(["unzip", "-oqq", self.base_directory_name +
                        ".zip"]) == 0, "ERROR UNZIPPING"

    def load_image_names(self):
        image_names_full_path = os.path.join(self.image_names_file)
        # This tends to interpret nan as a float
        self.image_data = pandas.read_csv(image_names_full_path, sep="\t",
                                          header=None).as_matrix()
        self.names = [str(i) for i in self.image_data[:, 1]]
        self.ids = [str(i) for i in self.image_data[:, 0]]

    def load_image_to_vector(self, img_path):
        img = image.load_img(img_path, target_size=self.target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = self.model.predict(x)

        return features.flatten()

    def load_image_data_from_labels(self, labels):
        base_path = os.path.join(self.base_directory_name,
                                 self.image_base_dir_rel_path)
        all_vectors = []
        for label in labels:
            file_path = get_file_path_from_label(label)
            vec = self.load_image_to_vector(file_path + "." +
                                            self.img_extension)
            all_vectors.append(vec)

        return all_vectors

    def save_all_data(self):
        self.load_image_names()
        filtered_names = \
            [x.upper() == self.label_for_training.upper() for x in self.names]
        assert sum(filtered_names) == 1, "ONE LABEL EXPECTED"
        label_ind = self.ids[self.names.index(self.label_for_training)]

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
        data_to_dump = np.array([all_pos_data, all_neg_data, all_pos_labels,
                                all_neg_labels])

        logging.debug("SAVING DATA")
        np.save(self.pickle_file, data_to_dump)
        return (all_pos_data, all_neg_data)


if __name__ == "__main__":
    try_keras = TryKerasConfig()
    try_keras.save_all_data()
