model.summary()
for layer in model.layers:
    print("input" + str(layer.input_shape)+"output" + str(layer.output_shape))
