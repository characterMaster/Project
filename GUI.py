import tensorflow as tf
from keras import backend as K
from tensorflow.keras import layers, models
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.15,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_set = "C:/Users/dell/Desktop/Compus_cars/train"
batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_set,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['MPV', 'sedan', 'hatchback', 'pickup', 'sports'],
    shuffle=True,
    seed=13,
    subset="training"
)

valid_generator = train_datagen.flow_from_directory(
    train_set,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['MPV', 'sedan', 'hatchback', 'pickup', 'sports'],
    shuffle=False,
    seed=13,
    subset="validation"
)

classes = train_generator.class_indices
classes_index = dict((v, k) for k, v in classes.items())
img = train_generator.filepaths
x_train, y_train = next(train_generator)


class ChannelAttention(layers.Layer):
    def __init__(self, channel, reduction=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        self.fc = self.build_fc(channel, reduction)
        self.softmax = layers.Activation('softmax')

    def build_fc(self, channel, reduction):
        input_tensor = layers.Input(shape=(channel,))
        x = layers.Dense(channel * reduction, use_bias=False)(input_tensor)
        x = layers.ReLU()(x)
        x = layers.Dense(channel, use_bias=False)(x)
        return models.Model(inputs=input_tensor, outputs=x)

    def call(self, x):
        y1 = self.avg_pool(x)
        y1 = tf.expand_dims(tf.expand_dims(self.fc(y1), 1), 1)

        y2 = self.max_pool(x)
        y2 = tf.expand_dims(tf.expand_dims(self.fc(y2), 1), 1)

        y = self.softmax(y1 + y2)
        return x + x * y


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "kernel_size = {}".format(kernel_size)
        padding = (kernel_size - 1) // 2
        self.conv = layers.Conv2D(1, kernel_size=kernel_size, padding='same', activation='softmax')

    def call(self, x):
        mask = self.conv(x)
        return x + x * mask


def WaveletTransformAxisY(batch_img):
    odd_img = batch_img[:, 0::2]
    even_img = batch_img[:, 1::2]
    L = (odd_img + even_img) / 2.0
    H = K.abs(odd_img - even_img)
    return L, H


def WaveletTransformAxisX(batch_img):
    # transpose + fliplr
    tmp_batch = K.permute_dimensions(batch_img, [0, 2, 1])[:, :, ::-1]
    _dst_L, _dst_H = WaveletTransformAxisY(tmp_batch)
    # transpose + flipud
    dst_L = K.permute_dimensions(_dst_L, [0, 2, 1])[:, ::-1, ...]
    dst_H = K.permute_dimensions(_dst_H, [0, 2, 1])[:, ::-1, ...]
    return dst_L, dst_H


def wavelet_transform(img):
    wavelet_L, wavelet_H = WaveletTransformAxisY(img)
    wavelet_LL, wavelet_LH = WaveletTransformAxisX(wavelet_L)
    wavelet_HL, wavelet_HH = WaveletTransformAxisX(wavelet_H)
    return wavelet_LL, wavelet_LH, wavelet_HL, wavelet_HH


def Wavelet(batch_image, n=4):
    # make channel first image
    batch_image = K.permute_dimensions(batch_image, [0, 3, 1, 2])
    r = batch_image[:, 0]
    g = batch_image[:, 1]
    b = batch_image[:, 2]

    wavelet_levels = []
    for i in range(n):  # 4 levels of decomposition
        r_wavelets = wavelet_transform(r)
        g_wavelets = wavelet_transform(g)
        b_wavelets = wavelet_transform(b)

        wavelet_data = list(r_wavelets) + list(g_wavelets) + list(b_wavelets)
        wavelet_level = K.stack(wavelet_data, axis=1)
        wavelet_levels.append(K.permute_dimensions(wavelet_level, [0, 2, 3, 1]))

        r, g, b = r_wavelets[0], g_wavelets[0], b_wavelets[0]  # LL for next level
    return wavelet_levels


def Wavelet_out_shape(input_shapes):
    base_size = 224
    return [tuple([None, base_size // (2 ** i), base_size // (2 ** i), 12]) for i in range(4)]


proposed = load_model('Res_DWAN.keras', safe_mode=False, custom_objects={'Wavelet': Wavelet,
                                                                         'Wavelet_out_shape': Wavelet_out_shape,
                                                                         'wavelet': layers.Lambda(Wavelet,
                                                                                                  Wavelet_out_shape,
                                                                                                  name='wavelet'),
                                                                         'ChannelAttention': ChannelAttention,
                                                                         'SpatialAttention': SpatialAttention})

with gr.Blocks() as demo:
    img_upload = gr.Image(type="pil")
    predict_button = gr.Button("Predict Image Category")
    clear_button = gr.Button("Clear")
    output_label = gr.Label()


    def predict_image(input_img):
        print("Loading model...")
        model = proposed
        img = input_img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction[0])
        return f"Predicted category: {predicted_label}"


    def clear():
        img_upload.clear()
        output_label.value = ""


    predict_button.click(predict_image, inputs=img_upload, outputs=output_label)
    clear_button.click(clear)

demo.queue()
demo.launch()
