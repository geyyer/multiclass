from datetime import datetime
import matplotlib.pyplot as plt
import gc
from keras import optimizers
from keras import layers
from keras import models


class ModelCustom:

    """model for multi-classification"""

    def __init__(self, src_size, dst_size):

        """
        constructor
        :param src_size: int - input size
        :param dst_size: int - output size
        """

        print('[CONSTRUCTOR] here it goes')
        self.model = None
        self.src_size = src_size
        self.dst_size = dst_size
        self.epochs = 0
        self.batch_size = 0
        self.mean_train_mae = [0]
        self.mean_val_mae = [0]

    def build(self, optimizer='SGD',
              lr=0.001, momentum=0.0, nesterov=False):

        """
        function builds inception net
        :param optimizer: string - optimizer
        :param lr: float - learning rate
        :param momentum: float - optimizer momentum
        :param nesterov: bool - nesterov momentum
        :return: model
        """

        # clean up
        self.model = None
        K.clear_session()
        gc.collect()

        print('[BUILD] building the inception model')
        height = self.src_size[0]
        width = self.src_size[1]
        channels = self.src_size[2]
        num_val = self.dst_size

        opts = {'SGD': optimizers.SGD,
                'RMSprop': optimizers.RMSprop,
                'Adagrad': optimizers.Adagrad,
                'Adadelta': optimizers.Adadelta,
                'Adam': optimizers.Adam,
                'Adamax': optimizers.Adamax,
                'Nadam': optimizers.Nadam
                }

        self.model = models.Sequential()

        self.model.add(layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=self.src_size))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv2D(32, kernel_size=3, activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.4))

        self.model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.4))

        self.model.add(layers.Conv2D(128, kernel_size=4, activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Flatten())
        self.model.add(layers.Dropout(0.4))
        self.model.add(layers.Dense(10, activation='softmax'))

        if optimizer == 'SGD':
            self.model.compile(optimizer=opts[optimizer](lr=lr,
                                                         momentum=momentum,
                                                         nesterov=nesterov
                                                         ),
                               loss='categorical_crossentropy',
                               metrics=['accuracy']
                               )
        else:
            self.model.compile(optimizer=opts[optimizer](lr=lr),
                               loss='categorical_crossentropy',
                               metrics=['accuracy']
                               )

        return self.model

    def selfie(self):

        """
        function prints model summary
        :return: None
        """

        print('model structure is')
        self.model.summary()

    def save(self):

        """
        function saves model structure and weights
        :return: None
        """

        # get comments
        comments = input('enter your comment ')

        # use date and time as name
        name = datetime.now()

        # save the model
        print('saving the model')
        model2json = self.model.to_json()
        with open('model.json', 'w') as json_file:
            json_file.write(model2json)

        # save the weights
        self.model.save_weights('model.h5')

        # save parameters and performance results
        with open('model.txt', 'a') as text_file:
            text_file.write('%s | %i | %i | %f | %f | %s \n' % (name,
                                                                self.epochs,
                                                                self.batch_size,
                                                                self.mean_train_mae[-1],
                                                                self.mean_val_mae[-1],
                                                                comments
                                                                ))

    def load(self):

        """
        function loads model structure and weights
        :return: model
        """

        # load the model
        print('loading the model')
        with open('model.json', 'r') as json_file:
            self.model = models.model_from_json(json_file.read())

        # load the weights
        self.model.load_weights('model.h5')

        # show summary
        self.model.summary()

        return self.model

    def train_model_holdout(self, train_x, train_y,
                            epochs, batch_size,
                            validation_data=None, data_size=-1):

        """
        function trains model and validates on a holdout
        :param train_x: numpy array - input data
        :param train_y: numpy array - labels
        :param epochs: int - number of epochs
        :param batch_size: int - training batch size
        :param validation_data: tuple of numpy arrays - (val_x, val_y)
        :param data_size: int - slice of input data to use
        :return: list - training and val metrics
        """

        print('training the model with holdout')

        self.epochs = epochs
        self.batch_size = batch_size

        history = self.model.fit(x=train_x[:data_size],
                                 y=train_y[:data_size],
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 validation_data=validation_data,
                                 verbose=1,
                                 )

        self.plot_results_holdout(history)

        return history

    @staticmethod
    def plot_results_holdout(history):

        """
        function plots training metrics
        :param history: list - training and val metrics
        :return: None
        """

        print(history.history)
        # Plot training & validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model error')
        plt.ylabel('Metric')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
