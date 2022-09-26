from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def DeepAE(x_train):
    encoding_dim = 128
    input_img = Input(shape=(1372,))

    # encoder layers
    encoded = Dense(512, activation='relu')(input_img)
    encoded = Dense(256, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)

    # decoder layers
    decoded = Dense(256, activation='relu')(encoder_output)
    decoded = Dense(512, activation='relu')(decoded)
    decoded = Dense(1372, activation='tanh')(decoded)

    # construct the autoencoder model
    autoencoder = Model(input=input_img, output=decoded)
    encoder = Model(input=input_img, output=encoder_output)

    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    # training
    # autoencoder_fit = autoencoder.fit(x_train, x_train, epochs=20, batch_size=128, shuffle=True)
    autoencoder.fit(x_train, x_train, epochs=20, batch_size=128, shuffle=True)

    # plotting
    encoded_imgs = encoder.predict(x_train)

    # ## 绘制迭代次数和loss之间的关系
    # ## 绘制图像
    # plt.figure(figsize=(10, 6.5))
    # plt.plot(autoencoder_fit.epoch, autoencoder_fit.history["loss"], "ro-", lw=2)
    #
    # x_major_locator = MultipleLocator(2)
    # # 把x轴的刻度间隔设置为2，并存在变量里
    # y_major_locator = MultipleLocator(0.0001)
    # # 把y轴的刻度间隔设置为0.0001，并存在变量里
    # ax = plt.gca()
    # # ax为两条坐标轴的实例
    # ax.xaxis.set_major_locator(x_major_locator)
    # # 把x轴的主刻度设置为2的倍数
    # ax.yaxis.set_major_locator(y_major_locator)
    # # 把y轴的主刻度设置为0.0001的倍数
    # plt.xlim(-1, 20)
    # plt.ylim(0.0001, 0.0008)
    # # plt.grid()
    # fontdict = {'color': 'black',
    #             'weight': 400,
    #             'size': 15}
    #
    # plt.xlabel("Epoch", fontdict=fontdict)
    # plt.ylabel("Model Loss", fontdict=fontdict)
    # plt.title("Deep Autoencoder", fontdict=fontdict)
    # plt.savefig('DeepAE model loss.png', dpi=300)
    # plt.show()

    return encoder_output, encoded_imgs

def DeepAE2(x_train):
    encoding_dim = 128
    input_img = Input(shape=(325,))

    # encoder layers
    encoded = Dense(250, activation='relu')(input_img)
    encoded = Dense(200, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)

    # decoder layers
    decoded = Dense(200, activation='relu')(encoder_output)
    decoded = Dense(250, activation='relu')(decoded)
    decoded = Dense(325, activation='tanh')(decoded)

    # construct the autoencoder model
    autoencoder = Model(input=input_img, output=decoded)
    encoder = Model(input=input_img, output=encoder_output)

    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    # training
    # autoencoder_fit = autoencoder.fit(x_train, x_train, epochs=20, batch_size=128, shuffle=True)
    autoencoder.fit(x_train, x_train, epochs=20, batch_size=128, shuffle=True)

    # plotting
    encoded_imgs = encoder.predict(x_train)

    # ## 绘制迭代次数和loss之间的关系
    # ## 绘制图像
    # plt.figure(figsize=(10, 6.5))
    # plt.plot(autoencoder_fit.epoch, autoencoder_fit.history["loss"], "ro-", lw=2)
    #
    # x_major_locator = MultipleLocator(2)
    # # 把x轴的刻度间隔设置为2，并存在变量里
    # y_major_locator = MultipleLocator(0.001)
    # # 把y轴的刻度间隔设置为0.0001，并存在变量里
    # ax = plt.gca()
    # # ax为两条坐标轴的实例
    # ax.xaxis.set_major_locator(x_major_locator)
    # # 把x轴的主刻度设置为2的倍数
    # ax.yaxis.set_major_locator(y_major_locator)
    # # 把y轴的主刻度设置为0.0001的倍数
    # plt.xlim(-1, 20)
    # plt.ylim(0, 0.008)
    # # plt.grid()
    # fontdict = {'color': 'black',
    #             'weight': 400,
    #             'size': 20}
    #
    # plt.xlabel("Epoch", fontdict=fontdict)
    # plt.ylabel("Model Loss", fontdict=fontdict)
    # plt.title("Deep Autoencoder", fontdict=fontdict)
    # plt.savefig('DeepAE model loss.png', dpi=300)
    # plt.show()

    return encoder_output, encoded_imgs

def transfer_label_from_prob(proba):
    label = [1 if val >= 0.5 else 0 for val in proba]
    return label


