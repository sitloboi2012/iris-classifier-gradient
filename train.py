import numpy as np
import pandas as pd
import sklearn
import onnxruntime as rt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = LogisticRegression()
clf.fit(X_train, y_train)
print(clf)

initial_type = [
    ('float_input', FloatTensorType([None, 4]))
]


# Export the model
onx = convert_sklearn(clf, initial_types=initial_type)
# Save it into wanted file
with open("iris_clf.onnx", "wb") as f:
    f.write(onx.SerializeToString())

sess = rt.InferenceSession("iris_clf.onnx")
input_name = sess.get_inputs()[0].name
pred_onx = sess.run(None, {input_name: X_test.astype(np.float32)})[0]
print(pred_onx)

sess = rt.InferenceSession("iris_clf.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run(
    [label_name], {input_name: X_test.astype(np.float32)})[0]
print(pred_onx)