print("Building libraries, waiting waiting ... please waiting ...")
import torch
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image
import torchvision.transforms as transforms

def main():
    def read_config():
        try:
            # print("Choose dataset (1: ph2 dataset, 2: kvasir dataset)")
            dataset_name = int(input("Choose dataset (1: ph2 dataset, 2: kvasir dataset): ").strip())
            if dataset_name not in [1, 2]:
                print("Invalid dataset choice.")
                return -1, "", ""

            # print("Enter one image path (image in the dataset selected above):")
            image_path = input("Enter one image path (image in the dataset selected above): ").strip()

            # print("Enter model weight path:")
            model_weight_path = input("Enter model weight path: ").strip()

            return dataset_name, image_path, model_weight_path
        except Exception as e:
            print(f"Error in read_config: {e}")
            return -1, "", ""
    def predict_one_ph2(model, data_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
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

            plt.savefig('output.png')
            # plt.show()
        except Exception as e:
            print(f"Error in predict_one_ph2: {e}")

    @torch.inference_mode()
    def predict_one_kvasir(model, image_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            kvasir_image = imread(image_path)
            image = Image.open(image_path).convert('RGB')
            image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.55714883, 0.32170294, 0.23581956),
                                     std=(0.31774015, 0.22082197, 0.18651856))
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

            plt.savefig('output.png')
            # plt.show()
        except Exception as e:
            print(f"Error in predict_one_kvasir: {e}")

    dataset_name, image_path, model_weight_path = read_config()

    if dataset_name != -1:
        try:
            # Predict
            if dataset_name == 1:
                ph2_model = torch.jit.load(model_weight_path)
                predict_one_ph2(ph2_model, image_path)
            elif dataset_name == 2:
                kvasir_model = torch.jit.load(model_weight_path)
                predict_one_kvasir(kvasir_model, image_path)
            else:
                print("Wrong dataset name")
        except Exception as e:
            print(f"Error in model prediction: {e}")
    else:
        print("Configuration was not successful.")

if __name__ == "__main__":
    main()