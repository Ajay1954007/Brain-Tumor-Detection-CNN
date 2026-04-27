# Brain Tumor Detection CNN

A CNN-based deep learning project for classifying brain MRI images into four categories:

- glioma_tumor
- meningioma_tumor
- no_tumor
- pituitary_tumor

The project uses TensorFlow/Keras to train a convolutional neural network and predict the tumor class for a given MRI image.

## Project Structure

```text
.
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
├── Testing/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
├── brain_tumor_model.h5
├── train.py
├── predict.py
├── test_model.py
├── check_tf.py
├── test.jpg
└── requirements.txt
```

## Requirements

- Python 3.12 recommended
- TensorFlow CPU
- Pillow


```




- The project should be run using the `.venv` Python environment.
- TensorFlow warning messages about CPU optimization or GPU support on Windows are informational and do not stop the project from running.
- The default Python on the system may not work if it is too new for TensorFlow, so Python 3.12 is recommended.
