import os
import pickle

# introdurre le stubs è utile per non dover rifare ogni volta il calcolo delle tracce
# aggiungendo dei checkpoints

def save_stubs(stub_path, object):
    if not os.path.exists(os.path.dirname(stub_path)):
        os.mkdir(os.path.dirname(stub_path))

    if stub_path is not None:
        with open(stub_path, 'wb') as f:
            pickle.dump(object, f)

def read_stubs(read_from_stub, stub_path):
    if read_from_stub and stub_path is not None and os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            object = pickle.load(f)
            return object
    return None