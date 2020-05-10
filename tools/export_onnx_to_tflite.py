import tensorflow as tf

graph_def_file = "/home/lh/pretrain-models/pose_higher_hrnet_256.pb"
input_arrays = ["input"]
output_arrays = ["outpu1","outpu2"]
converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("/home/lh/pretrain-models/pose_higher_hrnet_256.pb.tflite", "wb").write(tflite_model)