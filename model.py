from keras.models import Model
from keras.layers import Conv2D, Dropout, Conv2DTranspose, Add, Cropping2D, Input
from utils import get_segmentation_model
from VGG16 import get_vgg_encoder


def crop(o1, o2, i):
    o_shape2 = Model(i, o2).output_shape

    output_height2 = o_shape2[1]
    output_width2 = o_shape2[2]

    o_shape1 = Model(i, o1).output_shape
    output_height1 = o_shape1[1]
    output_width1 = o_shape1[2]

    cx = abs(output_width1 - output_width2)
    cy = abs(output_height2 - output_height1)

    if output_width1 > output_width2:
        o1 = Cropping2D(cropping=((0, 0), (0, cx)))(o1)
    else:
        o2 = Cropping2D(cropping=((0, 0), (0, cx)))(o2)

    if output_height1 > output_height2:
        o1 = Cropping2D(cropping=((0, cy), (0, 0)))(o1)
    else:
        o2 = Cropping2D(cropping=((0, cy), (0, 0)))(o2)

    return o1, o2


def fcn_8_vgg(n_classes, input_height=224, input_width=320, channels=3):
    img_input, levels = get_vgg_encoder(
        input_height=input_height, input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels

    o = f5

    o = Conv2D(4096, (7, 7), activation='relu', padding='same')(o)
    o = Dropout(0.5)(o)
    o = Conv2D(4096, (1, 1), activation='relu', padding='same')(o)
    o = Dropout(0.5)(o)

    o = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal')(o)
    o = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)

    o2 = f4
    o2 = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal')(o2)

    o, o2 = crop(o, o2, img_input)

    o = Add()([o, o2])

    o = Conv2DTranspose(n_classes, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(o)
    o2 = f3
    o2 = Conv2D(n_classes, (1, 1), kernel_initializer='he_normal')(o2)

    o2, o = crop(o2, o, img_input)
    o = Add(name="seg_feats")([o2, o])

    o = Conv2DTranspose(n_classes, kernel_size=(16, 16), strides=(8, 8), use_bias=False)(o)

    model = get_segmentation_model(img_input, o)
    model.model_name = "fcn_8_vgg"
    return model


if __name__ == '__main__':
    m = fcn_8_vgg(n_classes=27)
