from keras import backend as K

def extract_representations(model,samples):
    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functor = K.function([inp, K.learning_phase()], outputs )   # evaluation function
    layers_outs = functor([samples])
    return layers_outs