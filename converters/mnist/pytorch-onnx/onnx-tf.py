import onnx
from onnx_tf.backend import prepare
import numpy as np
from IPython.display import display
from PIL import Image

# Load the ONNX file
model = onnx.load('output/mnist.onnx')

# Import the ONNX model to Tensorflow
tf_rep = prepare(model)

# Input nodes to the model
print('inputs:', tf_rep.inputs)

# Output nodes from the model
print('outputs:', tf_rep.outputs)

# All nodes in the model
print('tensor_dict:')
print(tf_rep.tensor_dict)

print('Image 1:')
img = Image.open('assets/two.png').resize((28, 28)).convert('L')
display(img)
output = tf_rep.run(np.asarray(img, dtype=np.float32)[np.newaxis, np.newaxis, :, :])
print('The digit is classified as ', np.argmax(output))

print('Image 2:')
img = Image.open('assets/three.png').resize((28, 28)).convert('L')
display(img)
output = tf_rep.run(np.asarray(img, dtype=np.float32)[np.newaxis, np.newaxis, :, :])
print('The digit is classified as ', np.argmax(output))