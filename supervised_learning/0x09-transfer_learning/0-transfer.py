import tensorflow.keras as K
""" Classify CIFAR 10 Dataset using Keras applications KGG16 """


def get_cifar10():
    """ loads and preprocess cifar10 data"""
    (train_X, train_Y), (X, Y) = K.datasets.cifar10.load_data()
    # convert int to float and scale between 0 - 1 (max is 255)
    X = X.astype('float32') / 255
    train_X = train_X.astype('float32') / 255

    Y = K.utils.to_categorical(Y, 10)
    train_Y = K.utils.to_categorical(train_Y, 10)

    return train_X, train_Y, X, Y

def define_model(train_Y):
    """ define vgg16 model, with modifications """
    vgg16_app = K.applications.vgg16.VGG16(include_top=False,
                                           weights='imagenet',
                                           input_tensor=K.Input((32, 32, 3)),
                                           classes=train_Y.shape[1],
                                           pooling='avg')
    model = K.Sequential()
    model.add(vgg16_app)
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.Dense(10, activation='softmax'))
    opt = K.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    # get and scale data
    train_X, train_Y, X, Y = get_cifar10()

    # define model for training using vgg16, and a few added layers
    model = define_model(train_Y)

    # generate data, shift, and flip to avoid over fitting
    generator = K.preprocessing.image.ImageDataGenerator(width_shift_range=0.1,
                                                        height_shift_range=0.1,
                                                        horizontal_flip=True)
    generator.fit(train_X)
    t_data = generator.flow(train_X, train_Y, batch_size=64)
    # train model
    steps = int(train_X.shape[0] / 64)
    history = model.fit_generator(t_data,
                                steps_per_epoch=steps,
                                epochs=50,
                                validation_data=(X, Y),
                                verbose=1)
    # evaluate
    _x, acc = model.evaluate(X, Y, verbose=1)
    # save model to cifar10.h5
    model.save('cifar10.h5')
