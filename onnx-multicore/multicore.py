#!/usr/bin/python3
import onnx
import numpy
from skl2onnx.helpers import collect_intermediate_steps, compare_objects
from timeit import timeit
from time import time
import onnxruntime as rt
from onnxconverter_common.data_types import FloatTensorType
from skl2onnx import convert_sklearn, __version__
import numpy as np
import pandas as pd

from sklearn import datasets, __version__ as skl_version
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logistic = LogisticRegression()
pca = PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

digits = datasets.load_digits()
X_digits = digits.data[:1000]
y_digits = digits.target[:1000]

pipe.fit(X_digits, y_digits)


################# ONNX

initial_types = [('input', FloatTensorType((None, X_digits.shape[1])))]
model_onnx = convert_sklearn(pipe, initial_types=initial_types,
                             target_opset=12)

################ ONNX Session opts

opts = rt.SessionOptions()
opts.enable_profiling = True
opts.intra_op_num_threads = 8
opts.inter_op_num_threads = 1
opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

################

sess = rt.InferenceSession(model_onnx.SerializeToString(), sess_options=opts, providers=['CPUExecutionProvider'])
print("skl predict_proba")
print(pipe.predict_proba(X_digits[:2]))
start = time()
onx_pred = sess.run(None, {'input': X_digits[:2].astype(np.float32)})[1]
end = time()
df = pd.DataFrame(onx_pred)
print("onnx predict_proba")
print(df.values)
print("Time", end-start)



"""
opts = onnxruntime.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
ort_session = onnxruntime.InferenceSession('/path/to/model.onnx', sess_options=opts)
"""
