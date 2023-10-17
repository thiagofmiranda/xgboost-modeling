import json
import csv
import json

from xgboost import XGBClassifier


def make_prediction(x):
    # Load the saved model
    loaded_model = XGBClassifier()
    loaded_model.load_model("data/main_model.model")
    predictions_out = loaded_model.predict(x)
    print(predictions_out)
    # Make prediction
    dict_out = {}
    for count, value in enumerate(predictions_out):
        dict_out[count] = float(value)

    # Load json for decoding and decode the output
    with open('data/encoder.json') as json_file:
        data = json.load(json_file)

    # Returns the actual species name
    return data[str(int(dict_out[0]))]

def add_to_data(x1, x2, x3, x4, y):
    new_row = [float(x1), float(x2), float(x3), float(x4), str(y)]

    with open("data/iris.data", "a") as csv_file:
        writer_object = csv.writer(csv_file)
        writer_object.writerow(new_row)