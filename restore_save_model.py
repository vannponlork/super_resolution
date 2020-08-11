
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from keras.applications import VGG19
import Utils_model
from Utils_model import VGG_LOSS
from keras.applications.vgg19 import VGG19
from keras.models import Model

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import     build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter




image_height = 384
image_width = 384
image_h_lr = image_height /4
image_w_lr = image_width /4
amount = 2
batch_size = 2
image_folder = './train2019'
model_path = './model'


image_shape = (image_height,image_width, 3)

class VGG_LOSS(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):
    
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False
    
        return K.mean(K.square(model(y_true) - model(y_pred)))

loss = VGG_LOSS(image_shape)

image_shape = (image_height,image_width, 3)
K.set_learning_phase(0)
# "./model/gen_model3000.h5"
keras_model = load_model("./model/gen_model3000.h5", custom_objects={'vgg_loss': loss.vgg_loss})

# Define num_output. If you have multiple outputs, change this number accordingly
config = keras_model.get_config()
weights = keras_model.get_weights()

new_model = keras_model.from_config(config)
new_model.set_weights(weights)

new_model.summary()


export_path = './folder_to_export'
builder = saved_model_builder.SavedModelBuilder(export_path)

signature = predict_signature_def(inputs={'images': new_model.input},
                                  outputs={'scores': new_model.output})

with K.get_session() as sess:
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tag_constants.SERVING],
                                         signature_def_map={'predict': signature})
    builder.save()




# Define num_output. If you have multiple outputs, change this number accordingly
# num_output = 1
# name_output = ""
# output = [None] * num_output
# out_node_names = [None] * num_output
# for i in range(num_output):
#     out_node_names[i] = name_output + str(i)
#     output[i] = tf.identity(keras_model.outputs[i], name=out_node_names[i])
    
# sess = K.get_session()
# constant_graph = tf.graph_util.convert_variables_to_constants(
#     sess,
#     sess.graph.as_graph_def(),
#     out_node_names  # All other operations relying on this will also be saved
# )


# output_file = "./model/model_file.pb"
# with tf.gfile.GFile(output_file, "wb") as f:
#     f.write(constant_graph.SerializeToString())



# print("Converted model was saved as {}.".format(output_file))

#================================================================================
#================================================================================


# import sys
# from keras.models import load_model
# import tensorflow as tf
# from keras import backend as K
# from tensorflow.python.framework import graph_util
# from tensorflow.python.framework import graph_io
# from tensorflow.python.saved_model import signature_constants
# from tensorflow.python.saved_model import tag_constants


# K.set_learning_phase(0)
# K.set_image_data_format('channels_last')

# INPUT_MODEL = sys.argv[1]
# NUMBER_OF_OUTPUTS = 1
# OUTPUT_NODE_PREFIX = 'output_node'
# OUTPUT_FOLDER= 'frozen'
# OUTPUT_GRAPH = 'frozen_model.pb'
# OUTPUT_SERVABLE_FOLDER = sys.argv[2]
# INPUT_TENSOR = sys.argv[3]




# try:
#     model = load_model(INPUT_MODEL)
# except ValueError as err:
#     print('Please check the input saved model file')
#     raise err

# output = [None]*NUMBER_OF_OUTPUTS
# output_node_names = [None]*NUMBER_OF_OUTPUTS

# print(output_node_names)

# exit()


# for i in range(NUMBER_OF_OUTPUTS):
#     output_node_names[i] = OUTPUT_NODE_PREFIX+str(i)
#     output[i] = tf.identity(model.outputs[i], name=output_node_names[i])
# print('Output Tensor names: ', output_node_names)


# sess = K.get_session()
# try:
#     frozen_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_node_names)    
#     graph_io.write_graph(frozen_graph, OUTPUT_FOLDER, OUTPUT_GRAPH, as_text=False)
#     print(f'Frozen graph ready for inference/serving at {OUTPUT_FOLDER}/{OUTPUT_GRAPH}')
# except:
#     print('Error Occured')



# builder = tf.saved_model.builder.SavedModelBuilder(OUTPUT_SERVABLE_FOLDER)

# with tf.gfile.GFile(f'{OUTPUT_FOLDER}/{OUTPUT_GRAPH}', "rb") as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())

# sigs = {}
# OUTPUT_TENSOR = output_node_names
# with tf.Session(graph=tf.Graph()) as sess:
#     tf.import_graph_def(graph_def, name="")
#     g = tf.get_default_graph()
#     inp = g.get_tensor_by_name(INPUT_TENSOR)
#     out = g.get_tensor_by_name(OUTPUT_TENSOR[0] + ':0')

#     sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
#         tf.saved_model.signature_def_utils.predict_signature_def(
#             {"input": inp}, {"outout": out})

#     builder.add_meta_graph_and_variables(sess,
#                                          [tag_constants.SERVING],
#                                          signature_def_map=sigs)
#     try:
#         builder.save()
#         print(f'Model ready for deployment at {OUTPUT_SERVABLE_FOLDER}/saved_model.pb')
#         print('Prediction signature : ')
#         print(sigs['serving_default'])
#     except:
#         print('Error Occured, please checked frozen graph')


