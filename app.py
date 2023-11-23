import streamlit as st
from st_audiorec import st_audiorec
from pathlib import Path
import torch
import numpy as np
import soundfile as sf
import wave

from model import SpeakerEncoderInference, Synthesizer, VocoderInference, preprocess_wav


enc_model_fpath = Path("saved_models/default/encoder.pt")
syn_model_fpath = Path("saved_models/default/synthesizer.pt")
voc_model_fpath = Path("saved_models/default/vocoder.pt")
seed = None

## Load the models one by one.
print("Preparing the encoder, the synthesizer and the vocoder...")
encoder = SpeakerEncoderInference()
encoder.load_model(enc_model_fpath)
synthesizer = Synthesizer(syn_model_fpath)
vocoder = VocoderInference()
vocoder.load_model(voc_model_fpath)


st.title("ReVoice Demo")

options = ["Upload a .wav audio file", "Record your own audio"]
selected = st.selectbox("Choose an option", options)

with st.form("Form"):
    if selected == options[0]:
        input_audio = st.file_uploader("Upload your .wav audio file here")
    if selected == options[1]:
        st.write("Record your audio")
        audio_data = st_audiorec()

    st.markdown("---")
    text = st.text_input("Write a sentence")
    submit = st.form_submit_button("Generate")


def load_audio(input_audio_path, output_audio_path):
     with wave.open(input_audio_path, 'rb') as input_wave:
        # Get the parameters of the input WAV file
        params = input_wave.getparams()

        # Open the output WAV file for writing
        with wave.open(output_audio_path, 'wb') as output_wave:
            # Set the parameters for the output WAV file
            output_wave.setparams(params)

            # Read the audio data from the input file
            audio_data = input_wave.readframes(params.nframes)

            # Write the audio data to the output file
            output_wave.writeframes(audio_data)

if submit:
    audio_path = "audio.wav"
    if selected == options[0]:
        load_audio(input_audio, audio_path)
    if selected == options[1]:
        with open("audio.wav", 'wb') as file:
            file.write(audio_data)

    with st.spinner("Generating cloned audio"):        
        preprocessed_wav = preprocess_wav(audio_path)

        # Creating embeddings
        embed = encoder.embed_utterance(preprocessed_wav)

        # # If seed is specified, reset torch seed and force synthesizer reload
        if seed is not None:
            torch.manual_seed(seed)
            synthesizer = Synthesizer(syn_model_fpath)

        # Synthesizing mel spectrograms
        texts = [text]
        embeds = [embed]
        specs = synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]

        # # If seed is specified, reset torch seed and reload vocoder
        if seed is not None:
            torch.manual_seed(seed)
            vocoder.load_model(voc_model_fpath)

        # Generating waveform
        generated_wav = vocoder.infer_waveform(spec)
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
        generated_wav = preprocess_wav(generated_wav)

        filename = "demo_output.wav"
        sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
    st.write("")
    st.write("**Cloned Audio**")
    st.audio("demo_output.wav")
