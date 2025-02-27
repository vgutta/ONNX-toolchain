import onnxruntime as ort
import onnx
import numpy
from skl2onnx.helpers import collect_intermediate_steps, compare_objects
from timeit import timeit
import onnxruntime as rt
from onnxconverter_common.data_types import FloatTensorType
from skl2onnx import convert_sklearn, __version__
import numpy as np
import pandas as pd

from sklearn import datasets, __version__ as skl_version
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

EXECUTION_PROVIDER = 'CUDAExecutionProvider'

logistic = LogisticRegression()
pca = PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

digits = datasets.load_digits()
X_digits = digits.data[:1000]
y_digits = digits.target[:1000]

pipe.fit(X_digits, y_digits)

initial_types = [('input', FloatTensorType((None, X_digits.shape[1])))]
model_onnx = convert_sklearn(pipe, initial_types=initial_types,
                             target_opset=12)

sess = rt.InferenceSession(model_onnx.SerializeToString(), providers=[EXECUTION_PROVIDER])

onx_pred = sess.run(None, {'input': X_digits[:2].astype(np.float32)})[1]

# No exception so they are the same.
compare_objects(pipe.predict_proba(X_digits[:2]), onx_pred)

print("scikit-learn")
print(timeit("pipe.predict_proba(X_digits[:1])",
             number=10000, globals=globals()))
print("onnxruntime on ", EXECUTION_PROVIDER)
print(timeit("sess.run(None, {'input': X_digits[:1].astype(np.float32)})[1]",
             number=10000, globals=globals()))