from types import MethodType
from keras.models import Model
from keras.layers import Reshape, Activation
import keras.backend as K
#from train import train
#from predict import predict, predict_multiple, evaluate

def get_segmentation_model(input, output):
    img_input = input
    o = output

    o_shape = Model(img_input, o).output_shape
    i_shape = Model(img_input, o).input_shape

    output_height = o_shape[1]
    output_width = o_shape[2]
    n_classes = o_shape[3]
    
    o = Reshape((output_height * output_width, -1))(o)
    o = Activation('softmax')(o)

    model = Model(img_input, o)
    
    model.output_width = output_width
    model.output_height = output_height
    model.n_classes = n_classes
    model.input_height = i_shape[1]
    model.input_width = i_shape[2]
    model.model_name = ""

    #model.train = MethodType(train, model)
    #model.predict_segmentation = MethodType(predict, model)
    #model.predict_multiple = MethodType(predict_multiple, model)
    #smodel.evaluate_segmentation = MethodType(evaluate, model)

    return model
