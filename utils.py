from keras.preprocessing.image import *
from xml.dom import minidom as xml


smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


class PASCALVOCIterator(Iterator):
    """ Iterator based on original DirectoryIterator from keras library but apapted to
		process files organized like in PASCAL VOC manner
	# Arguments
		directory: Path to the root directory
			Root folder should has next structure:
			|-<root_directory>
				|-JPEGImages (contains images with .jpg extension
				|-ImageSets (contains .txt files with image sets (names of images without extension))
				|-Annotations (contains .xml files with annotation of images)
		target_file: File with current image set
		image_data_generator: Instance of `ImageDataGenerator`
			to use for random transformations and normalization.
		target_size: tuple of integers, dimensions to resize input images to.
		color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
		classes: List of strings with classes names
		class_mode: Mode for yielding the targets:
			`"binary"`: float targets,
			`"sparse"`: integer targets,
			`"input"`: targets are images identical to input images (mainly
				used to work with autoencoders),
			`None`: no targets get yielded (only input images are yielded).
		batch_size: Integer, size of a batch.
		shuffle: Boolean, whether to shuffle the data between epochs.
		seed: Random seed for data shuffling.
		data_format: String, one of `channels_first`, `channels_last`.
		save_to_dir: Optional directory where to save the pictures
			being yielded, in a viewable format. This is useful
			for visualizing the random transformations being
			applied, for debugging purposes.
		save_prefix: String prefix to use for saving sample
			images (if `save_to_dir` is set).
		save_format: Format to use for saving sample images
			(if `save_to_dir` is set).
		interpolation: Interpolation method used to resample the image if the
			target size is different from that of the loaded image.
			Supported methods are "nearest", "bilinear", and "bicubic".
			If PIL version 1.1.3 or newer is installed, "lanczos" is also
			supported. If PIL version 3.4.0 or newer is installed, "box" and
			"hamming" are also supported. By default, "nearest" is used.
	"""

    def __init__(self, directory, target_file, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='binary',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None, save_to_dir=None,
                 save_prefix='', save_format='png',
                 interpolation='nearest'):
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.target_file = target_file
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation

        # first, count the number of samples and classes
        self.samples = 0

        assert classes, "Provide correct classes"

        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        # Create paths to directories and check are their exist
        self.images_path = os.path.join(self.directory, "JPEGImages")
        self.imagesets_path = os.path.join(self.directory, "ImageSets")
        self.annotations_path = os.path.join(self.directory, "Annotations")

        self.image_set_file_path = os.path.join(self.imagesets_path, self.target_file)

        for item in [self.images_path, self.imagesets_path, self.annotations_path,
                     self.image_set_file_path]:
            assert os.path.exists(item), "Path does not exists: {0}".format(item)

        # read all files from target_file and form set
        with open(self.image_set_file_path) as f:
            content = f.readlines()

        self.image_set = [line.strip() for line in content if line is not "\n"]

        self.samples = len(self.image_set)

        print('Found %d images belonging to %d classes.' % (self.samples, self.num_classes))

        self.filenames = []
        self.classes = np.zeros((self.samples, self.num_classes), dtype='int32')

        for idx, fl in enumerate(self.image_set):
            self.filenames.append(os.path.join(self.images_path, fl + ".jpg"))
            annotation_path = os.path.join(self.annotations_path, fl + ".xml")
            objects = xml.parse(annotation_path).getElementsByTagName("object")
            for obj in objects:
                label = obj.getElementsByTagName("name")[0]
                deleted = obj.getElementsByTagName("deleted")
                if deleted:
                    if isinstance(deleted, list):
                        deleted = deleted[0]
                    if not int(deleted.firstChild.nodeValue):
                        self.classes[idx][self.class_indices[str(label.firstChild.nodeValue)]] = 1

        super(PASCALVOCIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fpath = self.filenames[j]
            img = load_img(fpath,
                           grayscale=grayscale,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=j,
                                                                  hash=np.random.randint(1e7),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        else:
            return batch_x
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
		# Returns
			The next batch.
		"""
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)