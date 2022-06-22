from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from non_pubmed_model_training import create_model, embed_text, train_general_model
from constants import global_constants
from extract_text import extract
import os

glob_c = global_constants()


def predictions(model_name: str, text_list: list):
    model = create_model()
    model.load_weights(glob_c.model_dir+model_name)
    x_pred = []
    y_pred = []
    x_pred, _ = embed_text(x_pred, y_pred, text_list, [])
    y_pred = model.predict(x_pred)
    return y_pred

def test_model(url: str):
  if "general_model_weights.index" not in os.listdir(glob_c.model_dir+"general_model/"):
    train_general_model()
  text_list = []
  extract([url], text_list)
  
  pred = predictions("general_model/general_model_weights", text_list)
  prediction_label = glob_c.false_news_label_name
  if pred[0][0]>0.208:
    prediction_label = glob_c.unsure_news_label_name
  elif pred[0][0]>0.008:
    prediction_label = glob_c.true_news_label_name
  print("prediction: ", prediction_label, "url: ", url, "pred: ", pred[0][0])
  return (url, prediction_label, pred[0][0])
