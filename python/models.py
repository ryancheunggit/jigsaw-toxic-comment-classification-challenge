#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ren Zhang @ ryanzjlib dot gmail dot com
from keras.layers import *
from .layers import *
from keras.models import Model
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from tensorflow.python.client import device_lib
from sklearn.metrics import roc_auc_score

# use cpu compatible rnn cells if on cpu only machine
cpu_only = len(device_lib.list_local_devices()) == 1
if cpu_only:
    lstm_cell = LSTM
    gru_cell = GRU
else:
    lstm_cell = CuDNNLSTM
    gru_cell = CuDNNGRU


def loss(y_true, y_pred):
    """ https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py """
    return K.binary_crossentropy(y_true, y_pred)


def fit_and_eval(
        model,
        x_train, y_train,
        x_valid, y_valid,
        batch_size=32,
        learning_rate_reducer_rounds=3,
        lr_decay_ratio=0.2,
        early_stopping_rounds=5,
        early_stopping_metric='auc'
    ):
    import keras.backend as K
    """ function to fit nn by epoch with early stopping and learning rate decay"""
    best_metric = -1
    best_weights = None
    best_epoch = 0
    current_epoch = 0
    current_batch_size = batch_size
    aucs = []
    losses = []
    while True:
        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=current_batch_size,
            epochs=1,
            verbose=1,
            validation_data=(x_valid, y_valid)
            )
        y_pred = model.predict(x_valid, batch_size=current_batch_size)
        # needed to prevent log_loss return nan, I had issue with default float32 predictions when using lstm
        # the y_pred values after sklearn's clipping code still sometimes got value 1 and causing log(1-1) return nan
        y_pred = y_pred.astype("float64")

        # === log loss is no longer the evaluation metric
        # losses = [log_loss(y_valid[:, j], y_pred[:, j]) for j in range(6)]
        # print(losses)
        # avg_losses = np.mean(losses)

        auc = roc_auc_score(y_valid, y_pred)
        loss = history.history['val_loss'][-1]
        aucs.append(auc)
        losses.append(loss)

        if early_stopping_metric == 'auc':
            current_metric = auc
        elif early_stopping_metric == 'loss':
            current_metric = loss

        print("current auc is : {}， current loss is {}".format(auc, loss))
        print("early stopping metric is {} - at epoch {} on validation with value : {}".format(
                early_stopping_metric, best_epoch, best_metric)
            )

        if early_stopping_metric == 'auc':
            if current_metric > best_metric or best_metric == -1:
                best_metric = current_metric
                best_weights = model.get_weights()
                best_epoch = current_epoch
        elif early_stopping_metric == 'loss':
            if current_metric < best_metric or best_metric == -1:
                best_metric = current_metric
                best_weights = model.get_weights()
                best_epoch = current_epoch

        if current_epoch - best_epoch == learning_rate_reducer_rounds:
            # learning rate decreasing
            current_lr = K.eval(model.optimizer.lr)
            new_lr = current_lr * lr_decay_ratio
            print("=== {} haven't improve for {} rounds, reducing learning rate to: {}".format(
                    early_stopping_metric,
                    learning_rate_reducer_rounds,
                    new_lr
                ))
            K.set_value(model.optimizer.lr, new_lr)
            current_batch_size //= 2
        if current_epoch - best_epoch == early_stopping_rounds:
            # early stopping
                    break

        current_epoch += 1

    print("auc at best epoch : {} - loss at best epoch : {}".format(aucs[best_epoch], losses[best_epoch]))
    model.set_weights(best_weights)
    return model, best_metric, best_epoch


def build_singlegru_model(
        embedding_matrix = None,
        embedding_trainable = False,
        embedding_dropout = False,
        max_sentence_length = None,
        max_features = None,
        num_recurrent_units = 64,
        num_cnn_filters = None,
        num_dense_units = 0,
        dropout_rate = 0.2,
        embeddings = None
    ):
    inputs = Input(shape=(max_sentence_length,))
    if embedding_matrix is None:
        embeddings = Embedding(input_dim = max_features, output_dim = 300)(inputs)
    else:
        embeddings = Embedding(
                input_dim = embedding_matrix.shape[0],
                output_dim = embedding_matrix.shape[1],
                weights = [embedding_matrix],
                trainable = embedding_trainable
            )(inputs)
    if embedding_dropout:
        embeddings = SpatialDropout1D(2 * dropout_rate)(embeddings)
    rnn = Bidirectional(gru_cell(num_recurrent_units, return_sequences=False))(embeddings)
    rnn = Dropout(dropout_rate)(rnn)
    if num_dense_units > 0:
        dense = Dense(num_dense_units, activation="relu")(rnn)
        outputs = Dense(6, activation="sigmoid")(dense)
    else:
        outputs = Dense(6, activation="sigmoid")(rnn)
    optimizer = Nadam(lr = 0.0005, clipvalue = 1, clipnorm = 1)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss = loss, optimizer = optimizer, metrics=['binary_accuracy'])
    return model

def build_1dcnn_model(
        embedding_matrix = None,
        max_sentence_length = 300,
        max_features = None,
        embedding_trainable = False,
        num_recurrent_units = None,
        num_cnn_filters = 128,
        kernel_size = 3,
        dropout_rate = 0.2,
        num_dense_units = 64,
        embedding_dropout = False,
        embeddings=None
    ):
    """
        1d 2 layer cnn with shortcuts
    """
    inputs = Input(shape=(max_sentence_length,))
    if embedding_matrix is None:
        embeddings = Embedding(input_dim = max_features, output_dim = 300)(inputs)
    else:
        embeddings = Embedding(
                input_dim = embedding_matrix.shape[0],
                output_dim = embedding_matrix.shape[1],
                weights = [embedding_matrix],
                trainable = embedding_trainable
            )(inputs)
    if embedding_dropout:
        embeddings = SpatialDropout1D(2 * dropout_rate)(embeddings)
    embedding_pool = GlobalMaxPooling1D()(embeddings)
    cnn = Conv1D(filters = 64, kernel_size = 3, padding = 'valid')(embeddings)
    # cnn = BatchNormalization()(cnn)
    cnn = Activation("relu")(cnn)
    cnn = Dropout(dropout_rate)(cnn)
    cnn1_pool = GlobalMaxPooling1D()(cnn)
    cnn = MaxPooling1D(2)(cnn)

    cnn = Conv1D(filters = 128, kernel_size = 3, padding = 'valid')(cnn)
    # cnn = BatchNormalization()(cnn)
    cnn = Activation("relu")(cnn)
    cnn = Dropout(dropout_rate)(cnn)
    cnn2_pool = GlobalMaxPooling1D()(cnn)
    cnn = MaxPooling1D(2)(cnn)

    cnn = Conv1D(filters = 256, kernel_size = 3, padding = 'valid')(cnn)
    # cnn = BatchNormalization()(cnn)
    cnn = Activation("relu")(cnn)
    cnn = Dropout(dropout_rate)(cnn)
    cnn3_pool = GlobalMaxPooling1D()(cnn)
    cnn = MaxPooling1D(2)(cnn)

    cnn = Conv1D(filters = 512, kernel_size = 3, padding = 'valid')(cnn)
    # cnn = BatchNormalization()(cnn)
    cnn = Activation("relu")(cnn)
    cnn = Dropout(dropout_rate)(cnn)
    cnn = GlobalMaxPooling1D()(cnn)

    concat = concatenate([embedding_pool, cnn1_pool, cnn2_pool, cnn3_pool, cnn])
    outputs = Dense(6, activation="sigmoid")(concat)
    optimizer = Adam(lr = 0.0005, clipvalue = 1, clipnorm = 1)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss = loss, optimizer = optimizer, metrics=['binary_accuracy'])
    return model

def build_2dcnn_model(
        embedding_matrix = None,
        max_sentence_length = 300,
        max_features = None,
        embedding_trainable = False,
        num_recurrent_units = None,
        num_cnn_filters = 64,
        filter_sizes = [1, 2, 3, 5], #,7],
        dropout_rate = 0.2,
        num_dense_units = 64,
        embedding_dropout = False,
        embeddings=None
    ):
    inputs = Input(shape=(max_sentence_length,))
    if embedding_matrix is None:
        embeddings = Embedding(input_dim = max_features, output_dim = 300)(inputs)
    else:
        embeddings = Embedding(
                input_dim = embedding_matrix.shape[0],
                output_dim = embedding_matrix.shape[1],
                weights = [embedding_matrix],
                trainable = embedding_trainable
            )(inputs)
    if embedding_dropout:
        embeddings = SpatialDropout1D(2 * dropout_rate)(embeddings)
    max_len, max_features = int(embeddings.shape[1]), int(embeddings.shape[2])
    cnn_in = Reshape((max_len, max_features, 1))(embeddings)

    cnn_outs = []
    for filter_size in filter_sizes:
        conv = Conv2D(num_cnn_filters, kernel_size=(filter_size, max_features), kernel_initializer='normal')(cnn_in)
        # conv = BatchNormalization()(conv)
        # conv = Dropout(dropout_rate / 2)(conv)
        conv = ELU()(conv)
        maxpool = GlobalMaxPooling2D()(conv)
        cnn_outs.append(maxpool)

    concat = Concatenate(axis=1)(cnn_outs)
    dense = Dropout(dropout_rate)(concat)
    outputs = Dense(6, activation="sigmoid")(dense)
    optimizer = Adam(lr = 0.0005, clipvalue = 1, clipnorm = 1)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss = loss, optimizer = optimizer, metrics=['binary_accuracy'])
    return model


def build_textcnn_model(
        embedding_matrix = None,
        max_features = None,
        max_sentence_length = None,
        embedding_trainable = False,
        dropout_rate = 0.2,
        embedding_dropout = False,
        num_recurrent_units = None,
        num_cnn_filters = None,
        embeddings=None
):
    inputs = Input(shape=(max_sentence_length,))
    if embedding_matrix is None:
        embeddings = Embedding(input_dim = max_features, output_dim = 300)(inputs)
    else:
        embeddings = Embedding(
                input_dim = embedding_matrix.shape[0],
                output_dim = embedding_matrix.shape[1],
                weights = [embedding_matrix],
                trainable = embedding_trainable
            )(inputs)
    if embedding_dropout:
        embeddings = SpatialDropout1D(2 * dropout_rate)(embeddings)
    kernel_sizes = [2, 4, 6]
    cnn_features = []
    for kernel_size in kernel_sizes:
        x = Conv1D(filters = 32, kernel_size = kernel_size, padding = 'valid')(embeddings)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size = 2)(x)

        x = Conv1D(filters = 64, kernel_size = kernel_size, padding = 'valid')(x)
        # x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size = 2)(x)
        # x = Flatten()(x)
        # x = Conv1D(filters = 32, kernel_size = kernel_size, padding = 'valid')(x)
        # x = Activation('relu')(x)
        # x = MaxPooling1D(pool_size = 2)(x)

        x = Flatten()(x)
        cnn_features.append(x)

    dense = Concatenate()(cnn_features)
    dense = Dropout(dropout_rate)(dense)
    dense = Dense(128, activation = "relu")(dense)
    dense = Dropout(dropout_rate)(dense)
    # dense = Dense(128, activation = "relu")(dense)
    # dense = Dropout(0.1)(dense)
    outputs = Dense(6, activation = "sigmoid")(dense)
    model = Model(inputs = inputs, outputs = outputs)
    model.summary()
    optimizer = Adam(lr=0.001, clipvalue = 1, clipnorm = 1)
    model.compile(optimizer=optimizer, loss=loss, metrics=["binary_accuracy"])
    return model

def build_vggcnn_model(
        embedding_matrix = None,
        embedding_trainable = False,
        max_sentence_length = None,
        max_features = None,
        num_filters = [16, 32, 64, 128],
        cnn_kernel_size = 3,
        cnn_strides = 1,
        cnn_pool_size = 2,
        dropout_rate = 0.2,
        embedding_dropout = False,
        num_recurrent_units = None,
        num_cnn_filters = None,
        embeddings=None
    ):
    inputs = Input(shape=(max_sentence_length,))
    if embedding_matrix is None:
        embeddings = Embedding(input_dim = max_features, output_dim = 300)(inputs)
    else:
        embeddings = Embedding(
                input_dim = embedding_matrix.shape[0],
                output_dim = embedding_matrix.shape[1],
                weights = [embedding_matrix],
                trainable = embedding_trainable
            )(inputs)
    if embedding_dropout:
        embeddings = SpatialDropout1D(2 * dropout_rate)(embeddings)
    for i in range(len(num_filters)):
        if i == 0:
            cnn = Conv1D(filters = num_filters[i], kernel_size = cnn_kernel_size, strides = cnn_strides, padding = "same")(embeddings)
        else:
            cnn = Conv1D(filters = num_filters[i], kernel_size = cnn_kernel_size, strides = cnn_strides, padding = "same")(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = Activation("relu")(cnn)
        cnn = Conv1D(filters = num_filters[0], kernel_size = cnn_kernel_size, strides = cnn_strides, padding = "same")(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = Activation("relu")(cnn)
        cnn = MaxPooling1D(pool_size = cnn_pool_size)(cnn)

    global_max_pool = GlobalMaxPooling1D()(cnn)
    global_avg_pool = GlobalAveragePooling1D()(cnn)
    concat = Concatenate()([global_max_pool, global_avg_pool])
    dense = Dense(128, activation = 'relu')(concat)
    dense = Dropout(dropout_rate)(dense)
    outputs = Dense(6, activation="sigmoid")(dense)
    optimizer = SGD(lr = 0.001, decay = 1e-6, momentum = 0.9, nesterov = True)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss = loss, optimizer = optimizer, metrics=['accuracy'])
    return model


def build_dpcnn_model(
        embedding_matrix = None,
        max_sentence_length = None,
        max_features = None,
        embedding_trainable = False,
        num_filters = [64, 64, 64, 64, 64, 64, 64],
        top_k = 4,
        dropout_rate = 0.2,
        num_recurrent_units = None,
        num_cnn_filters = None,
        embedding_dropout = False,
        embeddings=None
    ):
    inputs = Input(shape=(max_sentence_length,))
    if embedding_matrix is None:
        embeddings = Embedding(input_dim = max_features, output_dim = 300)(inputs)
    else:
        embeddings = Embedding(
                input_dim = embedding_matrix.shape[0],
                output_dim = embedding_matrix.shape[1],
                weights = [embedding_matrix],
                trainable = embedding_trainable
            )(inputs)

    if embedding_dropout:
        embeddings = SpatialDropout1D(dropout_rate)(embeddings)
    for layer_num, num_filter in enumerate(num_filters):
        if layer_num == 0:
            cnn = Conv1D(filters = num_filter, kernel_size = 3, strides = 1, padding = 'same')(embeddings)
        else:
            cnn = Conv1D(filters = num_filter, kernel_size = 3, strides = 1, padding = 'same')(cnn_out)
        cnn = BatchNormalization()(cnn)
        cnn = Dropout(dropout_rate / 2)(cnn)
        cnn = Activation("relu")(cnn)
        cnn = Conv1D(filters = num_filter, kernel_size = 3, strides = 1, padding = 'same')(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = Dropout(dropout_rate / 2)(cnn)
        cnn = Activation("relu")(cnn)
        if layer_num == 0:
            short_cut = Conv1D(filters = num_filter, kernel_size=1, padding='same')(embeddings)
            short_cut = Activation("relu")(short_cut)
            cnn = Dropout(dropout_rate / 2)(cnn)
            cnn_out = Add()([cnn, short_cut])
        else:
            cnn_out = Add()([cnn, cnn_out])
        if layer_num != (len(num_filters) - 1):
            cnn_out = MaxPooling1D(pool_size=3, strides=2)(cnn_out)

    # def _top_k(x, top_k = 4):
    #     x = tf.transpose(x, [0, 2, 1])
    #     k_max = tf.nn.top_k(x, k = top_k)
    #     return tf.reshape(k_max[0], (-1, num_filters[-1] * top_k))
    #
    # k_max = Lambda(_top_k, output_shape = (num_filters[-1] * top_k, ))(cnn)
    # dense = Dense(512, activation = "relu", kernel_initializer = "he_normal")(k_max)
    # dense = Dropout(dropout_rate)(dense)
    # dense = Dense(512, activation = "relu", kernel_initializer = "he_normal")(dense)
    # dense = Dropout(dropout_rate)(dense)
    cnn_out = GlobalMaxPooling1D()(cnn_out)
    # cnn_out = Flatten()(cnn_out)
    dense = Dense(256)(cnn_out)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)
    dense = Dropout(2*dropout_rate)(dense)
    outputs = Dense(6, activation="sigmoid")(dense)
    optimizer = Adam(lr = 0.001, decay = 1e-6)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss = loss, optimizer = optimizer, metrics=['accuracy'])
    return model


def build_dualgru_model(
        embedding_matrix = None,
        embedding_trainable = False,
        embedding_dropout = False,
        max_sentence_length = None,
        max_features = None,
        num_recurrent_units = 64,
        num_cnn_filters = None,
        num_dense_units = 32,
        dropout_rate = 0.2,
        embeddings=None
    ):
    inputs = Input(shape=(max_sentence_length,))
    if embedding_matrix is None:
        embeddings = Embedding(input_dim = max_features, output_dim = 300)(inputs)
    else:
        embeddings = Embedding(
                input_dim = embedding_matrix.shape[0],
                output_dim = embedding_matrix.shape[1],
                weights = [embedding_matrix],
                trainable = embedding_trainable
            )(inputs)
    if embedding_dropout:
        embeddings = SpatialDropout1D(2 * dropout_rate)(embeddings)
    rnn = Bidirectional(gru_cell(num_recurrent_units, return_sequences=True))(embeddings)
    rnn = SpatialDropout1D(dropout_rate)(rnn)
    rnn = Bidirectional(gru_cell(num_recurrent_units, return_sequences=False))(rnn)
    rnn = Dropout(dropout_rate)(rnn)
    if num_dense_units > 0:
        dense = Dense(num_dense_units, activation="relu")(rnn)
        outputs = Dense(6, activation="sigmoid")(dense)
    else:
        outputs = Dense(6, activation="sigmoid")(rnn)
    optimizer = RMSprop(lr = 0.001, clipvalue = 1, clipnorm = 1, decay = 1e-6)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss = loss, optimizer = optimizer, metrics=['binary_accuracy'])
    return model

def build_duallstm_model(
        embedding_matrix = None,
        embedding_trainable = False,
        embedding_dropout = False,
        max_sentence_length = None,
        max_features = None,
        num_recurrent_units = 54,
        num_cnn_filters = None,
        num_dense_units = 32,
        dropout_rate = 0.2,
        embeddings=None
    ):
    inputs = Input(shape=(max_sentence_length,))
    if embedding_matrix is None:
        embeddings = Embedding(input_dim = max_features, output_dim = 300)(inputs)
    else:
        embeddings = Embedding(
                input_dim = embedding_matrix.shape[0],
                output_dim = embedding_matrix.shape[1],
                weights = [embedding_matrix],
                trainable = embedding_trainable
            )(inputs)
    if embedding_dropout:
        embeddings = SpatialDropout1D(2 * dropout_rate)(embeddings)
    rnn = Bidirectional(lstm_cell(num_recurrent_units, return_sequences = True))(embeddings)
    rnn = SpatialDropout1D(dropout_rate)(rnn)
    rnn = Bidirectional(lstm_cell(num_recurrent_units, return_sequences = False))(rnn)
    rnn = Dropout(dropout_rate)(rnn)
    if num_dense_units > 0:
        dense = Dense(num_dense_units, activation="relu")(rnn)
        outputs = Dense(6, activation="sigmoid")(dense)
    else:
        outputs = Dense(6, activation="sigmoid")(rnn)
    optimizer = RMSprop(lr = 0.001, clipvalue = 1, clipnorm = 1)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss = loss, optimizer = optimizer, metrics=['binary_accuracy'])
    return model


def build_lstmpool_model(
        embedding_matrix = None,
        embedding_trainable = False,
        embedding_dropout = False,
        max_sentence_length = None,
        max_features = None,
        num_recurrent_units = 64,
        num_cnn_filters = None,
        num_dense_units = 32,
        dropout_rate = 0.2,
        embeddings=None
    ):
    inputs = Input(shape=(max_sentence_length,))
    if embedding_matrix is None:
        embeddings = Embedding(input_dim = max_features,output_dim = 300)(inputs)
    else:
        embeddings = Embedding(
                input_dim = embedding_matrix.shape[0],
                output_dim = embedding_matrix.shape[1],
                weights = [embedding_matrix],
                trainable = embedding_trainable
            )(inputs)
    if embedding_dropout:
        embeddings = SpatialDropout1D(2 * dropout_rate)(embeddings)
    rnn = Bidirectional(lstm_cell(num_recurrent_units, return_sequences = True))(embeddings)
    rnn = SpatialDropout1D(dropout_rate)(rnn)
    global_max_pool = GlobalMaxPooling1D()(rnn)
    global_avg_pool = GlobalAveragePooling1D()(rnn)
    concat = concatenate([global_max_pool, global_avg_pool])
    outputs = Dense(6, activation="sigmoid")(concat)
    optimizer = RMSprop(lr = 0.001, clipvalue = 1, clipnorm = 1, decay = 1e-6)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss = loss, optimizer = optimizer, metrics=['binary_accuracy'])
    return model


def build_grupool_model(
        embedding_matrix = None,
        embedding_trainable = False,
        embedding_dropout = False,
        max_sentence_length = None,
        max_features = None,
        num_recurrent_units = 72,
        num_cnn_filters = None,
        num_dense_units = 32,
        dropout_rate = 0.,
        embeddings = None
    ):
    inputs = Input(shape=(max_sentence_length,))
    if embedding_matrix is None:
        embeddings = Embedding(input_dim = max_features, output_dim = 300)(inputs)
    else:
        embeddings = Embedding(
            input_dim = embedding_matrix.shape[0],
            output_dim = embedding_matrix.shape[1],
            weights = [embedding_matrix],
            trainable = embedding_trainable
        )(inputs)
    if embedding_dropout:
        embeddings = SpatialDropout1D(2 * dropout_rate)(embeddings)
    rnn = Bidirectional(lstm_cell(num_recurrent_units, return_sequences = True))(embeddings)
    rnn = SpatialDropout1D(dropout_rate)(rnn)
    global_max_pool = GlobalMaxPooling1D()(rnn)
    global_avg_pool = GlobalAveragePooling1D()(rnn)
    concat = concatenate([global_max_pool, global_avg_pool])
    outputs = Dense(6, activation="sigmoid")(concat)
    optimizer = RMSprop(lr = 0.001, clipvalue = 1, clipnorm = 1)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss = loss, optimizer = optimizer, metrics=['binary_accuracy'])
    return model


def build_grucnn_model(
        embedding_matrix = None,
        embedding_trainable = False,
        embedding_dropout = False,
        max_sentence_length = None,
        max_features = None,
        num_recurrent_units = 128,
        num_cnn_filters = 64,
        dropout_rate = 0.2,
        embeddings=None
    ):
    inputs = Input(shape=(max_sentence_length,))
    if embedding_matrix is None:
        embeddings = Embedding(input_dim = max_features, output_dim = 300)(inputs)
    else:
        embeddings = Embedding(
                input_dim = embedding_matrix.shape[0],
                output_dim = embedding_matrix.shape[1],
                weights = [embedding_matrix],
                trainable = embedding_trainable
            )(inputs)
    if embedding_dropout:
        embeddings = SpatialDropout1D(2 * dropout_rate)(embeddings)
    rnn = Bidirectional(gru_cell(num_recurrent_units, return_sequences = True))(embeddings)
    # rnn = SpatialDropout1D(dropout_rate)(rnn)
    cnn = Conv1D(filters = num_cnn_filters, kernel_size = 3, padding = 'valid', kernel_initializer = 'he_uniform')(rnn)
    global_max_pool = GlobalMaxPooling1D()(cnn)
    global_avg_pool = GlobalAveragePooling1D()(cnn)
    concat = concatenate([global_max_pool, global_avg_pool])
    outputs = Dense(6, activation="sigmoid")(concat)
    optimizer = Adam(lr = 0.001, clipvalue = 1, clipnorm = 1)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile( loss = loss, optimizer = optimizer, metrics=['binary_accuracy'])
    return model


def build_lstmattention_model(
        embedding_matrix = None,
        embedding_trainable = False,
        embedding_dropout = False,
        max_sentence_length = None,
        max_features = None,
        num_recurrent_units = 64,
        num_cnn_filters = None,
        num_dense_units = 32,
        dropout_rate = 0.2,
        embeddings = None
    ):
    inputs = Input(shape=(max_sentence_length,))
    if embedding_matrix is None:
        embeddings = Embedding( input_dim = max_features, output_dim = 300)(inputs)
    else:
        embeddings = Embedding(
                input_dim = embedding_matrix.shape[0],
                output_dim = embedding_matrix.shape[1],
                weights = [embedding_matrix],
                trainable = embedding_trainable
            )(inputs)

    if embedding_dropout:
        embeddings = SpatialDropout1D(2 * dropout_rate)(embeddings)
    rnn = Bidirectional(lstm_cell(num_recurrent_units, return_sequences = True))(embeddings)
    rnn = SpatialDropout1D(dropout_rate)(rnn)
    attention = AttentionWeightedAverage()(rnn)
    dense = Dense(num_dense_units, activation="relu")(attention)
    outputs = Dense(6, activation="sigmoid")(dense)
    optimizer = RMSprop(lr = 0.0005, clipvalue = 1, clipnorm = 1)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile( loss = loss, optimizer = optimizer, metrics=['binary_accuracy'])
    return model