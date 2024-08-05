# SOC Final Project Report

Created time: August 5, 2024 8:25 PM
Reviewed: No

# Problem Statement:

Fine-grained classification:
Train a CNN model with an upper limit of 10M parameters on the CUB_200_2011.tgz dataset downloaded from the following URL. [https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1)

Link to my GitHub Repo [here](https://github.com/Harshvardhan-10/SOC-Deep-Learning/tree/Fine_Grain_CNN)

# Model Architecture

This model uses the base layers from a pretrained model called EfficientNetV2. GitHub Repo [here](https://github.com/google/automl/tree/master/efficientnetv2)

- Input Layer with input shape(224,224,3)
- A Custom Augmentation Layer, to Augment the images
- Lambda Layer, preprocess the input images using the preprocess_input function of MobileNetV2
- Base Model using the base layers of the pre-trained MobileNetV2 model
- Global Average Pooling, Convert CNN layers to fully connected layer, It computes the average of all pixels in each feature map and returns that as a single element
- Dense layer with 1024 units, 'ReLu' activation
- Dropout Layer
- Dense Layer with 512 units, 'ReLu' activation
- Dropout Layer
- Dense Layer with 200 units, 'softmax' activation

```python
model = models.Sequential([
    layers.Input(shape=input_shape),
    CustomAugment(),
    layers.Lambda(preprocess_input),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
```

![Untitled](SOC%20Final%20Report%20ce6f62e8e1634c0b9f47357b1e29c6b7/Untitled.png)

## Training Details:

Used the suggested train_test_split provided with the dataset.

I applied the bounding boxes provided with the dataset to crop the images to focus on the subject.

![Before applying bounding Box](SOC%20Final%20Report%20ce6f62e8e1634c0b9f47357b1e29c6b7/Untitled.jpeg)

Before applying bounding Box

![After applying Bounding Box](SOC%20Final%20Report%20ce6f62e8e1634c0b9f47357b1e29c6b7/Untitled%201.png)

After applying Bounding Box

<aside>
➡️ Train samples: 5994
Test samples: 5794
Steps per epoch: 188
Validation steps: 182

</aside>

<aside>
➡️ Total number of Parameters: 4,197,128
Trainable Parameters: 1,939,144
Non-trainable Parameters: 2,257,984

</aside>

<aside>
➡️ Batch Size = 32
Epochs trained = 36, total epochs were 50, early_stopping stopped it at 36
Optimizer used = Adam(lr = 0.001)

</aside>

<aside>
➡️ Total Run Time: 673.3s - GPU T4 x2

</aside>

<aside>
💡 Train Accuracy: 68.20%
Test Accuracy: 63.08%

</aside>

### Model Accuracy and Model Loss curves:

![Untitled](SOC%20Final%20Report%20ce6f62e8e1634c0b9f47357b1e29c6b7/Untitled%202.png)


### Best Model
The model with best val_accuracy is uploaded to the GitHub Repo linked at the start of this document.  
And also uploaded to GDrive [here](https://drive.google.com/file/d/1a1a5jLwrsgY9kwwzVIZKl3vesG_oLZkV/view?usp=sharing) 
