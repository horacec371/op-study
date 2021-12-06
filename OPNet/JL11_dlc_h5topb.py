import os
import argparse
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from JL11_dlc_model import get_model

def h5_to_pb(pb_save_name):
  try:
    img_shape = (12, 128, 256)
    num_classes = 6
    model = get_model(img_shape, num_classes)
    model.load_weights('./saved_model/JL11_dlc_model.h5')
    model.summary()
  except ValueError:
    print("----- Error: You use 'json mode' to covert this model")
    return 1

  full_model = tf.function(lambda Input: model(Input))
  full_model = full_model.get_concrete_function([tf.TensorSpec(model.inputs[i].shape, model.inputs[i].dtype)
                                                 for i in range(len(model.inputs))])

  # Get frozen Concrete Function
  frozen_func = convert_variables_to_constants_v2(full_model)
  print('----- Frozen pb model graph: ')
  print(frozen_func.graph)
  layers = [op.name for op in frozen_func.graph.get_operations()]
  print("-" * 50)
  print("----- Frozen pb model inputs needed for converting pb to dlc: ")
  print(frozen_func.inputs)
  print("----- Frozen pb model outputs needed for converting pb to dlc: ")
  print(frozen_func.outputs)

  tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir="./saved_model",
                    name=pb_save_name,
                    as_text=False)
  print("----- OK: pb is saved in ./saved_model ")

if __name__=="__main__":
   parser = argparse.ArgumentParser(description='h5 to pb')
   parser.add_argument('pb', type=str, default="transition", help='save pb name')
   args = parser.parse_args()
   h5_to_pb(args.pb)
