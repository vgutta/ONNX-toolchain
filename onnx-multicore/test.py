#!/usr/bin/python3
import onnxruntime

opts = onnxruntime.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
ort_session = onnxruntime.InferenceSession('/path/to/model.onnx', sess_options=opts)
