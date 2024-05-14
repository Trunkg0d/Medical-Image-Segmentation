# Medical Image Segmentation

## Description
In modern medicine, medical image segmentation plays a crucial role in diagnosing and treating diseases. Recent studies have proposed numerous medical image segmentation models, ranging from traditional methods to deep learning approaches. This project surveys and proposes a model using the U-Net architecture, combining techniques in image processing and deep learning to improve the accuracy of medical image segmentation.

## Getting Started

### Download model weights
First, we need to download the model weights from [this link](https://studenthcmusedu-my.sharepoint.com/:f:/g/personal/21120157_student_hcmus_edu_vn/EvvTe_Uf1cZKrpeATLmUCokBlswlVG7qFNuAyubDQNOsZw?e=PC5tL8), There are two pre-trained models available:

- Ternausnet 11
    - Ternausnet 11 ph2 (for the PH2 dataset)
    - Ternausnet 11 kvasir (for the Kvasir dataset)
- Ternausnet 16
    - Ternausnet 16 ph2 (for the PH2 dataset)
    - Ternausnet 16 kvasir (for the Kvasir dataset)

### Installing
Download the project from GitHub either as a zip folder or by executing the following command:
```
git clone https://github.com/Trunkg0d/Medical-Image-Segmentation.git
```

### Run program
First, from the root folder, find the directory containing the main.py file (it in source folder) and navigate to that directory. Then, open the terminal and run the following command:
```
python main.py
```
The program will prompt you to select the dataset you want to test (PH2 or Kvasir) by pressing 1 (PH2) or 2 (Kvasir). Next, you need to enter the path of an image from the chosen dataset (the image must be from the dataset you selected to produce accurate results). Finally, the program will ask for the path to the model you downloaded earlier (e.g., Ternausnet_11_ph2.pt for the Ternausnet 11 model for the PH2 dataset).
## Authors

Contributors names and contact info

[@Trunkg0d](https://www.facebook.com/htak2003)
[@LeVanTan]()
[@NguyenQuangVinh]()

## License

## Acknowledgments
Eternal gratitude to the wizards and sorceresses of open-source magic.
