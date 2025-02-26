# AI Image and Text Generator using Streamlit and TensorFlow

## Overview
This project provides two AI-powered applications using **Streamlit** and **TensorFlow**:
1. **AI Object Recognition:** Uses ResNet50 to classify objects in an uploaded image.
2. **AI Text Generation:** Uses OpenAI's GPT to generate text based on user input.

## Features
- Upload an image and get top-3 classification predictions.
- Enter a text prompt to generate AI-powered text.
- Uses pre-trained deep learning models.

## Installation
Ensure you have Python installed, then install the required dependencies:

```bash
pip install streamlit tensorflow openai pillow numpy torch diffusers transformers
```

## Running the Application
To start the Streamlit app, run the following command:

```bash
streamlit run your_script.py
```

Replace `your_script.py` with the actual filename.

## Usage
### AI Object Recognition:
1. Upload an image in **JPG, PNG, or JPEG** format.
2. The model analyzes and predicts the top three objects in the image.

### AI Text Generation:
1. Enter a text prompt.
2. The AI generates a response using GPT.

## Dependencies
- Python
- Streamlit
- TensorFlow
- OpenAI API (for text generation)
- Pillow
- NumPy
- Torch & Diffusers (for image generation if needed)

## Notes
- The object recognition model used is **ResNet50**, trained on ImageNet.
- OpenAI API requires an API key for text generation.
- This project is for educational and demonstration purposes.

## License
This project is open-source and available for educational purposes.

