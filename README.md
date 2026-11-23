# DermaScan AI

A Streamlit app that classifies a photo of a skin lesion and explains the result in plain language.

It uses two models:

1. **Computer vision**: a Hugging Face image classifier ([`Anwarkh1/Skin_Cancer-Image_Classification`](https://huggingface.co/Anwarkh1/Skin_Cancer-Image_Classification)), trained on the HAM10000 dataset's 7 lesion types, predicts the condition and a confidence score.
2. **Gemini**: turns that prediction into a short, calm explanation with suggested next steps.

> **Disclaimer:** For educational purposes only. This is not a medical device and does not give a medical diagnosis. See a doctor about any skin concern.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then enter your Gemini API key in the sidebar (get one at [aistudio.google.com](https://aistudio.google.com/)), upload a JPG or PNG of the lesion, and click **Analyze Lesion**.

The first run downloads the classifier model from Hugging Face, so it takes a little longer.
