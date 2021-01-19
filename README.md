A reproduction of an issue with using tf2onnx on a model produced by tensorflow recommenders.

Train the example tfrs model from the [tfrs docs](https://www.tensorflow.org/recommenders/examples/basic_retrieval):

```
pipenv install
pipenv run python src/user_movies.py tmp/saved-model
```

Then, (attempt to) convert it to onnx:

```
pipenv run python -m tf2onnx.convert --saved-model tmp/saved-model --output tmp/model.onnx
```

Which outputs:

```
2021-01-19 13:51:37,636 - INFO - Using tensorflow=2.4.0, onnx=1.8.0, tf2onnx=1.8.1/4e49f3
2021-01-19 13:51:37,636 - INFO - Using opset <onnx, 9>
2021-01-19 13:51:37,636 - WARNING - Shape of placeholder unknown is unknown, treated it as a scalar
2021-01-19 13:51:37,644 - WARNING - Cannot infer shape for StatefulPartitionedCall/brute_force/sequential/string_lookup/None_lookup_table_find/LookupTableFindV2: StatefulPartitionedCall/brute_force/sequential/string_lookup/None_lookup_table_find/LookupTableFindV2:0
2021-01-19 13:51:37,644 - WARNING - Cannot infer shape for StatefulPartitionedCall/brute_force/sequential/embedding/embedding_lookup: StatefulPartitionedCall/brute_force/sequential/embedding/embedding_lookup:0
2021-01-19 13:51:37,644 - WARNING - Cannot infer shape for StatefulPartitionedCall/brute_force/sequential/embedding/embedding_lookup/Identity: StatefulPartitionedCall/brute_force/sequential/embedding/embedding_lookup/Identity:0
2021-01-19 13:51:37.653819: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-01-19 13:51:37.653851: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-01-19 13:51:37.653857: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      
2021-01-19 13:51:37.654754: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:196] None of the MLIR optimization passes are enabled (registered 0 passes)
2021-01-19 13:51:37,656 - INFO - Computed 1 values for constant folding
2021-01-19 13:51:37,685 - INFO - folding node using tf type=Identity, name=Func/StatefulPartitionedCall/input/_5
Traceback (most recent call last):
  File "/usr/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/home/mbp/.virtualenvs/tfrs-onnx-repro-oWWnfm9c/lib/python3.8/site-packages/tf2onnx/convert.py", line 198, in <module>
    main()
  File "/home/mbp/.virtualenvs/tfrs-onnx-repro-oWWnfm9c/lib/python3.8/site-packages/tf2onnx/convert.py", line 165, in main
    g = process_tf_graph(tf_graph,
  File "/home/mbp/.virtualenvs/tfrs-onnx-repro-oWWnfm9c/lib/python3.8/site-packages/tf2onnx/tfonnx.py", line 491, in process_tf_graph
    fold_constants_using_tf(g, outputs_to_values, outputs_to_dtypes)
  File "/home/mbp/.virtualenvs/tfrs-onnx-repro-oWWnfm9c/lib/python3.8/site-packages/tf2onnx/tfonnx.py", line 52, in fold_constants_using_tf
    ops[idx] = g.make_const(new_node_name, val)
  File "/home/mbp/.virtualenvs/tfrs-onnx-repro-oWWnfm9c/lib/python3.8/site-packages/tf2onnx/graph.py", line 562, in make_const
    onnx_tensor = numpy_helper.from_array(np_val, name)
  File "/home/mbp/.virtualenvs/tfrs-onnx-repro-oWWnfm9c/lib/python3.8/site-packages/onnx/numpy_helper.py", line 112, in from_array
    raise NotImplementedError(
NotImplementedError: ('Unrecognized object in the object array, expect a string, or array of bytes: ', "<class 'bytes'>")
```

