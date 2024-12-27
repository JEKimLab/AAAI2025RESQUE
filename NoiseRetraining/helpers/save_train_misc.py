import os, sys
import pickle
import numpy as np

def output_to_std_and_file(directory, file_name, data):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file_name)
    print(data)
    with open(file_path, "a") as file:
        file.write(data)


def save_numpy_to_csv(directory, file_name, data):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file_name)
    np.savetxt(file_path, data, delimiter=',')


def save_model(std_string, model, save_to, note=None):
    output_to_std_and_file(save_to, "standard_output.txt", std_string)

    save_model = {
        "model" : model,
        "note" : note
    }
    save_model_details = os.path.join(save_to, "model.pkl")
    with open(save_model_details, "wb") as fp:   
        pickle.dump(save_model, fp)


def save_img(trainset_copy, save_to, type_samples, index_positions):
    dir_path = os.path.join(save_to, "samples", type_samples)
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass
    for i, index in enumerate(index_positions):
        image, label = trainset_copy[index]  
        image_path = os.path.join(dir_path, f"image_{index}.png")
        image.save(image_path) 