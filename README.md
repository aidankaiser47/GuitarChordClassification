# GuitarChordClassification

Main work shown in Guitar_Chord_Classifier.ipynb

Utilizing multiple neural networks to classify guitar chords

Uses a public data set from Hugging Face: https://huggingface.co/datasets/severyn-k/isolated-guitar-chords

Trains my own CNN based on the spectrogram images of the audio files
Uses pretrained ResNet-18 to also try classifying guitar chords
Also utilizes a GradCAM model to try and explain what my personal CNN is looking at.