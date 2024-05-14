import torch
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def read_config():
    print("Choose dataset (1: ph2 dataset, 2: kvasir dataset)")
    dataset_name = int(input())
    if dataset_name != 1 and dataset_name != 2:
        print("Wrong dataset")
        return -1, -1, -1

    print("Enter one image path (image in the dataset selected above):")
    image_path = str(input())

    print("Enter model weight path:")
    model_weight_path = str(input())
    return dataset_name, image_path, model_weight_path

def predict_one_ph2(model, data_path):
    size = (224, 224)
    ph2_image = imread(data_path)
    ph2_image = resize(ph2_image, size, mode='constant', anti_aliasing=True)

    np_ph2_image = np.expand_dims(ph2_image, axis=0)

    test_dataset = np.rollaxis(np_ph2_image, 3, 1)
    model.eval()  # testing mode

    with torch.no_grad():
        X = torch.FloatTensor(test_dataset).to(device)
        pred = (torch.sigmoid(model(X)).squeeze() > 0.5).to(int).cpu().numpy()

    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(1, 2, 1)
    plt.imshow(ph2_image)
    plt.title("Original image")

    fig.add_subplot(1, 2, 2)
    plt.imshow(pred, cmap="gray")
    plt.title("Predicted")

    plt.show()

@torch.inference_mode()
def predict_one_kvasir(model, image_path):
    kvasir_image = imread(image_path)
    image = Image.open(image_path).convert('RGB')
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.55714883, 0.32170294, 0.23581956), std=(0.31774015, 0.22082197, 0.18651856))
    ])
    image = image_transform(image)

    model.eval()
    with torch.no_grad():
        image = image[None, ...].to(device)

        output = model(image)

        predict_label = (torch.sigmoid(output.squeeze()) > 0.5).to(int)

    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(1, 2, 1)
    plt.imshow(kvasir_image)
    plt.title("Original image")

    fig.add_subplot(1, 2, 2)
    plt.imshow(predict_label.cpu().numpy(), cmap="gray")
    plt.title("Predicted")

    plt.show()

dataset_name, image_path, model_weight_path = read_config()

if dataset_name != -1:
    # Predict
    if dataset_name == 1:
        ph2_model = torch.jit.load(model_weight_path)
        predict_one_ph2(ph2_model, image_path)
    elif dataset_name == 2:
        kvasir_model = torch.jit.load(model_weight_path)
        predict_one_kvasir(kvasir_model, image_path)
    else:
        print("Wrong dataset name")