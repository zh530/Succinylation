import os
os.environ["CUDA_VISIBLE_DEVICES"]="0";
import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.allow_growth = True

import numpy as np
import time
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn import metrics
from keras.optimizers import Adam
from keras.layers import Input, Conv1D, AveragePooling1D, GlobalAveragePooling1D, Dropout, Flatten, Dense, \
     Activation, Concatenate, Reshape, GlobalMaxPooling1D, Add, Permute, multiply, Lambda, Conv2D
from keras.models import Model
from keras.regularizers import l2
from keras.utils import to_categorical
from scipy import interp


def conv_factory(x, concat_axis, filters, dropout_rate=None, weight_decay=1e-4):
    """x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)"""
    x = Activation('elu')(x)
    x = Conv1D(filters=filters,
               kernel_size=3,
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


def transition(x, concat_axis, filters, dropout_rate=None, weight_decay=1e-4):
    """x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)"""
    x = Activation('elu')(x)
    x = Conv1D(filters=filters,
               kernel_size=1,
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling1D(pool_size=2, strides=2)(x)
    return x


def denseblock(x, concat_axis, layers, filters, growth_rate, dropout_rate=None, weight_decay=1e-4):
    list_feature_map = [x]
    for i in range(layers):
        x = conv_factory(x, concat_axis, growth_rate,
                         dropout_rate, weight_decay)
        list_feature_map.append(x)
        x = Concatenate(axis=concat_axis)(list_feature_map)
        filters = filters + growth_rate
    return x, filters


def channel_attention(input_feature, ratio=16):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                         kernel_initializer='he_normal',
                         activation='relu',
                         use_bias=True,
                         bias_initializer='zeros')

    shared_layer_two = Dense(channel,
                         kernel_initializer='he_normal',
                         use_bias=True,
                         bias_initializer='zeros')

    avg_pool = GlobalAveragePooling1D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling1D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('hard_sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
       cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7
    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          activation='hard_sigmoid',
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def cbam_block(cbam_feature, ratio=16):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in CBAM: Convolutional Block Attention Module.
    """
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


# focal_loss
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss0(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(1e-8 + pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + 1e-8))
    return focal_loss0

# categorical_focal_loss
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss1(y_true, y_pred):
        # Define epsilon so that the backpropagation will not K_FOLD in NaN
        epsilon = 1e-7
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        alpha1 = tf.where(tf.equal(y_true, [1.0, 0.0]), y_true * (1.0 - alpha), y_true * alpha)
        weight = alpha1 * y_true * K.pow((K.ones_like(y_pred) - y_pred), gamma)
        loss = weight * cross_entropy
        loss = K.sum(loss, axis=1)
        return loss
    return focal_loss1

# binary_focal_loss
def binary_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha_t*((1-p_t)^gamma)*log(p_t)
        p_t = y_pred, if y_true = 1
        p_t = 1-y_pred, otherwise
        alpha_t = alpha, if y_true=1
        alpha_t = 1-alpha, otherwise
        cross_entropy = -log(p_t)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss2(y_true, y_pred):
        # Define epsilon so that the backpropagation will not K_FOLD in NaN
        epsilon = 1e-7
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        loss = weight * cross_entropy
        loss = K.sum(loss, axis=1)
        return loss
    return focal_loss2


def build_model(windows=16, concat_axis=-1, denseblocks=3, layers=3, filters=96,
                growth_rate=32, dropout_rate=0.2, weight_decay=1e-4):
    input_1 = Input(shape=(2*windows+1, 21))     
    input_2 = Input(shape=(2*windows+1, 5))      
    input_3 = Input(shape=(2*windows+1, 8))      
    x_1 = Conv1D(filters=filters, kernel_size=3,
                 kernel_initializer="he_normal",
                 padding="same", use_bias=False,
                 kernel_regularizer=l2(weight_decay))(input_1)
    # Add denseblocks
    filters_1 = filters
    for i in range(denseblocks - 1):
        x_1, filters_1 = denseblock(x_1, concat_axis=concat_axis, layers=layers,
                                    filters=filters_1, growth_rate=growth_rate,
                                    dropout_rate=dropout_rate, weight_decay=weight_decay)

        x_1 = transition(x_1, concat_axis=concat_axis, filters=filters_1,
                         dropout_rate=dropout_rate, weight_decay=weight_decay)

    # The last denseblock
    x_1, filters_1 = denseblock(x_1, concat_axis=concat_axis, layers=layers,
                                filters=filters_1, growth_rate=growth_rate,
                                dropout_rate=dropout_rate, weight_decay=weight_decay)
    """x_1 = BatchNormalization(axis=concat_axis,
                             gamma_regularizer=l2(weight_decay),
                             beta_regularizer=l2(weight_decay))(x_1)"""
    x_1 = Activation('elu')(x_1)
    x_1 = cbam_block(x_1)


    ##################################################################################
    x_2 = Conv1D(filters=filters, kernel_size=3,
                 kernel_initializer="he_normal",
                 padding="same", use_bias=False,
                 kernel_regularizer=l2(weight_decay))(input_2)
    # Add denseblocks
    filters_2 = filters
    for i in range(denseblocks - 1):
        x_2, filters_2 = denseblock(x_2, concat_axis=concat_axis, layers=layers,
                                    filters=filters_2, growth_rate=growth_rate,
                                    dropout_rate=dropout_rate, weight_decay=weight_decay)
        x_2 = transition(x_2, concat_axis=concat_axis, filters=filters_2,
                         dropout_rate=dropout_rate, weight_decay=weight_decay)

    # The last denseblock
    x_2, filters_2 = denseblock(x_2, concat_axis=concat_axis, layers=layers,
                                filters=filters_2, growth_rate=growth_rate,
                                dropout_rate=dropout_rate, weight_decay=weight_decay)
    """x_2 = BatchNormalization(axis=concat_axis,
                             gamma_regularizer=l2(weight_decay),
                             beta_regularizer=l2(weight_decay))(x_2)"""
    x_2 = Activation('elu')(x_2)
    x_2 = cbam_block(x_2)


    #################################################################################
    x_3 = Conv1D(filters=filters, kernel_size=3,
                     kernel_initializer="he_normal",
                     padding="same", use_bias=False,
                     kernel_regularizer=l2(weight_decay))(input_3)
    # Add denseblocks
    filters_3 = filters
    for i in range(denseblocks - 1):
        x_3, filters_3 = denseblock(x_3, concat_axis=concat_axis, layers=layers,
                                        filters=filters_3, growth_rate=growth_rate,
                                        dropout_rate=dropout_rate, weight_decay=weight_decay)
        x_3 = transition(x_3, concat_axis=concat_axis, filters=filters_3,
                             dropout_rate=dropout_rate, weight_decay=weight_decay)
        # The last denseblock
    x_3, filters_3 = denseblock(x_3, concat_axis=concat_axis, layers=layers,
                                    filters=filters_3, growth_rate=growth_rate,
                                    dropout_rate=dropout_rate, weight_decay=weight_decay)
    """x_3 = BatchNormalization(axis=concat_axis,
                                 gamma_regularizer=l2(weight_decay),
                                 beta_regularizer=l2(weight_decay))(x_3)"""
    x_3 = Activation('elu')(x_3)
    x_3 = cbam_block(x_3)


    #################################################################################
    x = Concatenate(axis=-1)([x_1, x_2, x_3])
    x = Flatten()(x)

    x = Dense(units=2, activation="softmax", use_bias=False,
              kernel_initializer='he_normal',
              kernel_regularizer=l2(weight_decay))(x)

    model = Model(inputs=[input_1, input_2, input_3], outputs=[x], name="DenseBlock")

    optimizer = Adam(lr=1e-4, epsilon=1e-8)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


# Performance evaluation function
def perform_eval(predictions, Y_test, verbose=0):

    class_label = np.uint8(np.argmax(predictions, axis=1))
    R = np.asarray(np.uint8([sublist[1] for sublist in Y_test]))

    CM = metrics.confusion_matrix(R, class_label, labels=None)
    CM = np.double(CM)  # CM[0][0]：TN，CM[0][1]：FP，CM[1][0]：FN，CM[1][1]：TP


    sn = (CM[1][1]) / (CM[1][1] + CM[1][0])  
    sp = (CM[0][0]) / (CM[0][0] + CM[0][1])  
    acc = (CM[1][1] + CM[0][0]) / (CM[1][1] + CM[0][0] + CM[0][1] + CM[1][0])  
    pre = (CM[1][1]) / (CM[1][1] + CM[0][1])  
    f1 = (2 * CM[1][1]) / (2 * CM[1][1] + CM[0][1] + CM[1][0])  
    mcc = (CM[1][1] * CM[0][0] - CM[0][1] * CM[1][0]) / np.sqrt((CM[1][1] + CM[0][1]) * (CM[1][1] + CM[1][0]) * (CM[0][0] + CM[0][1]) * (CM[0][0] + CM[1][0]))  # (TP*TN-FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))^1/2
    gmean = np.sqrt(sn * sp)
    auroc = metrics.roc_auc_score(y_true=R, y_score=np.asarray(predictions)[:, 1], average="macro")
    aupr = metrics.average_precision_score(y_true=R, y_score=np.asarray(predictions)[:, 1], average="macro")

    if verbose == 1:
        print("Sn(Recall):", "{:.4f}".format(sn), "Sp:", "{:.4f}".format(sp), "Acc:", "{:.4f}".format(acc),
              "Pre(PPV):", "{:.4f}".format(pre), "F1:", "{:.4f}".format(f1), "MCC:", "{:.4f}".format(mcc),
              "G-mean:", "{:.4f}".format(gmean), "AUROC:", "{:.4f}".format(auroc), "AUPR:", "{:.4f}".format(aupr))

    return [sn, sp, acc, pre, f1, mcc, gmean, auroc, aupr]


# save results
def write_res(filehandle, res, fold=0):
    filehandle.write("Fold: " + str(fold) + " ")
    filehandle.write("Sn(Recall): %s Sp: %s Acc: %s Pre(PPV): %s F1: %s MCC: %s G-mean: %s AUROC: %s AUPR: %s\n" %
                     ("{:.4f}".format(res[0]),
                      "{:.4f}".format(res[1]),
                      "{:.4f}".format(res[2]),
                      "{:.4f}".format(res[3]),
                      "{:.4f}".format(res[4]),
                      "{:.4f}".format(res[5]),
                      "{:.4f}".format(res[6]),
                      "{:.4f}".format(res[7]),
                      "{:.4f}".format(res[8]))
                     )
    filehandle.flush()
    return

# loss-epoch，acc-epoch
def figure(history, K_FOLD, fold):
    """
    iters = np.arange(len(history_dict["loss"]))
    plt.figure()
    # acc
    plt.plot(iters, history_dict["acc"], color='r', label='train acc')
    # loss
    plt.plot(iters, history_dict["loss"], color='g', label='train loss')
    # val_acc
    plt.plot(iters, history_dict["val_acc"], color='b', label='val acc')
    # val_loss
    plt.plot(iters, history_dict["val_loss"], color='k', label='val loss')
    plt.grid(True) # 设置网格线
    plt.xlabel('epochs')
    plt.ylabel('loss-acc')
    plt.legend(loc="upper right") #设置图例位置
    plt.savefig("./%d折交叉第%d折.png" % (K_FOLD, fold))  # 保存图片
    """
    def show_train_history(train_history, train_metrics, validation_metrics):
        plt.plot(train_history.history[train_metrics])
        plt.plot(train_history.history[validation_metrics])
        plt.title('Train History')
        plt.grid(True)  
        plt.ylabel(train_metrics)
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left') 

    def plt_fig(history, K_FOLD, fold):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        show_train_history(history, 'acc', 'val_acc')
        plt.subplot(1, 2, 2)
        show_train_history(history, 'loss', 'val_loss')
        plt.savefig("./picture/%d折交叉第%d折-elu.png" % (K_FOLD, fold))  
        plt.close()

    return plt_fig(history, K_FOLD, fold)


if __name__ == '__main__':

    BATCH_SIZE = 2000
    K_FOLD = 10
    N_EPOCH = 2000
    WINDOWS = 16


    res_file = open("./result/cv/train.txt", "w", encoding='utf-8')
    res = []
    tprs = []
    aurocs = []
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(12, 4))

    for fold in range(K_FOLD):

        f_r_train = open("./dataset/train/%s/cv_10/Succinylation_Pos_Neg_train-%d.txt" % (str(2*WINDOWS+1), fold), "r", encoding='utf-8')
        f_r_test = open("./dataset/train/%s/cv_10/Succinylation_Pos_Neg_val-%d.txt" % (str(2*WINDOWS+1), fold), "r", encoding='utf-8')

        train_data = f_r_train.readlines()

        test_data = f_r_test.readlines()

        f_r_train.close()
        f_r_test.close()


        from information_coding_e1 import one_hot, Phy_Chem_Inf, Structure_Inf

        train_X_1, train_Y = one_hot(train_data, windows=WINDOWS)
        train_Y = to_categorical(train_Y, num_classes=2)
        test_X_1, test_Y = one_hot(test_data, windows=WINDOWS)
        test_Y = to_categorical(test_Y, num_classes=2)

        train_X_2 = Phy_Chem_Inf(train_data, windows=WINDOWS)
        test_X_2 = Phy_Chem_Inf(test_data, windows=WINDOWS)
        train_X_3 = Structure_Inf(train_data, windows=WINDOWS)
        test_X_3 = Structure_Inf(test_data, windows=WINDOWS)


        model = build_model(windows=WINDOWS)
        model.summary()

        print("fold:", str(fold))
        history = model.fit(x=[train_X_1, train_X_2, train_X_3], y=train_Y, batch_size=BATCH_SIZE, epochs=N_EPOCH, shuffle=True, class_weight={0: 1.0, 1: 10.9}, callbacks=[EarlyStopping(monitor='val_loss', patience=20, mode='auto')], verbose=2, validation_data=([test_X_1, test_X_2, test_X_3], test_Y))

        predictions = model.predict(x=[test_X_1, test_X_2, test_X_3], verbose=0)

        res = perform_eval(predictions, test_Y, verbose=1)

        write_res(res_file, res, fold)

        figure(history, K_FOLD, fold)

        R = np.asarray(np.uint8([sublist[1] for sublist in test_Y]))
        plt.subplot(1, 2, 1)
        fpr, tpr, auc_thresholds = metrics.roc_curve(y_true=R, y_score=np.asarray(predictions)[:, 1], pos_label=1)
        auroc_score = metrics.auc(fpr, tpr)
        aurocs.append(auroc_score)
        interp_tpr = interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.4f)' % (fold, auroc_score))

        plt.subplot(1, 2, 2)
        precision, recall, pr_thresholds = metrics.precision_recall_curve(y_true=R, probas_pred=np.asarray(predictions)[:, 1], pos_label=1)
        aupr_score = metrics.auc(recall, precision)

        plt.plot(recall, precision, lw=1, alpha=0.3, label='PR fold %d(area=%0.4f)' % (fold, aupr_score))

        model.save('./model/fold%d.h5' % fold)


    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auroc = metrics.auc(mean_fpr, mean_tpr)
    std_auroc = np.std(aurocs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auroc, std_auroc), lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontdict={'family': 'Times New Roman', 'size': 10})
    plt.ylabel('True Positive Rate', fontdict={'family': 'Times New Roman', 'size': 10})
    plt.title('Receiver operating characteristic curves', fontsize=10)
    plt.legend(loc="lower right")


    plt.subplot(1, 2, 2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall', fontdict={'family': 'Times New Roman', 'size': 10})
    plt.ylabel('Precision', fontdict={'family': 'Times New Roman', 'size': 10})
    plt.title('Precision-Recall curves', fontsize=10)
    plt.legend(loc="upper right")
    plt.savefig("./result/cv/%d折交叉.png" % (K_FOLD))
    plt.close()


    res_file.close()
    ##########################################################################################
    time.sleep(1)
    sn = []
    sp = []
    acc = []
    pre = []
    f1 = []
    mcc = []
    gmean = []
    auroc = []
    aupr = []

    f_r = open("./result/cv/train.txt", "r", encoding='utf-8')
    lines = f_r.readlines()

    for line in lines:
        x = line.split()
        sn.append(float(x[3]))
        sp.append(float(x[5]))
        acc.append(float(x[7]))
        pre.append(float(x[9]))
        f1.append(float(x[11]))
        mcc.append(float(x[13]))
        gmean.append(float(x[15]))
        auroc.append(float(x[17]))
        aupr.append(float(x[19]))

    mean_sn = np.mean(sn)
    mean_sp = np.mean(sp)
    mean_acc = np.mean(acc)
    mean_pre = np.mean(pre)
    mean_f1 = np.mean(f1)
    mean_mcc = np.mean(mcc)
    mean_gmean = np.mean(gmean)
    mean_auroc = np.mean(auroc)
    mean_aupr = np.mean(aupr)


    print("mean_sn:", "{:.4f}".format(mean_sn), "mean_sp:", "{:.4f}".format(mean_sp), "mean_acc:", "{:.4f}".format(mean_acc),
          "mean_pre:", "{:.4f}".format(mean_pre), "mean_f1:", "{:.4f}".format(mean_f1), "mean_mcc", "{:.4f}".format(mean_mcc),
          "mean_gmean:", "{:.4f}".format(mean_gmean), "mean_auroc:", "{:.4f}".format(mean_auroc), "mean_aupr:", "{:.4f}".format(mean_aupr))

    f_r.close()
