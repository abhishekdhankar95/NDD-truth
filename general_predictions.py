from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from non_pubmed_model_training import create_model, embed_text
from constants import global_constants

glob_c = global_constants()


def predictions(model_name: str, text_list: list):
    model = create_model()
    model.load_weights(glob_c.model_dir+model_name)
    x_pred = []
    y_pred = []
    embed_text(x_pred, y_pred, text_list, [])
    y_pred = model.predict()
    return y_pred

'''
def test(url):
    text_list = []
    extract([url], text_list)
    pred = predictions("general_model", text_list)
    return pred[0]
'''
    
