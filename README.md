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

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you are setting up the project from scratch, create a virtual environment first:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Check TensorFlow

```powershell
.\.venv\Scripts\python.exe check_tf.py
```

Expected output:

```text
Start
TensorFlow OK
```

## Train the Model

```powershell
.\.venv\Scripts\python.exe train.py
```

This trains the CNN using images from the `Training` folder and validates using the `Testing` folder. After training, the model is saved as:

```text
brain_tumor_model.h5
```

## Test Model Loading

```powershell
.\.venv\Scripts\python.exe test_model.py
```

## Predict an Image

Place the image as `test.jpg` in the project folder, then run:

```powershell
.\.venv\Scripts\python.exe predict.py
```

Example output:

```text
Prediction: no_tumor
```

## Notes

- The project should be run using the `.venv` Python environment.
- TensorFlow warning messages about CPU optimization or GPU support on Windows are informational and do not stop the project from running.
- The default Python on the system may not work if it is too new for TensorFlow, so Python 3.12 is recommended.
