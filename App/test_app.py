import tempfile

import torch
import gradio as gr
from diffusers import StableDiffusionPipeline
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import torchaudio
import numpy as np #for tupel transformation in musicgen
import soundfile as sf


# ----------- MODEL SETUP -----------

# Stable Diffusion (Text-to-Image)
sd_model_id = "runwayml/stable-diffusion-v1-5"
pipe_sd = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16)
pipe_sd = pipe_sd.to("cuda")

# MusicGen (Text-to-Music)
musicgen_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
musicgen_model.to("cuda")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

# ----------- INFERENCE FUNCTIONS -----------

def generate_image(prompt):
    with torch.autocast("cuda"):
        image = pipe_sd(prompt, height=512, width=512, num_inference_steps=25).images[0]
    return image

#def generate_music(prompt):
#    inputs = processor(text=[prompt], return_tensors="pt").to("cuda")
#    audio_values = musicgen_model.generate(**inputs, max_new_tokens=256)
#    audio_values = audio_values.cpu().squeeze(0)
#    sample_rate = musicgen_model.config.audio_encoder.sampling_rate
#    return (sample_rate, audio_values)

'''
def generate_music(prompt):
    inputs = processor(text=[prompt], return_tensors="pt").to("cuda")
    audio_values = musicgen_model.generate(**inputs, max_new_tokens=256)
    audio_tensor = audio_values[0].cpu()

    # tupel transformation (Normierung und Umwandlung)
    audio_numpy = audio_tensor.numpy()
    audio_numpy = audio_numpy / np.max(np.abs(audio_numpy))  # Normalisieren auf [-1, 1]

    sample_rate = musicgen_model.config.audio_encoder.sampling_rate
    return (sample_rate, audio_numpy)

def generate_music_with_download(prompt):
    inputs = processor(text=[prompt], return_tensors="pt").to("cuda")
    audio_values = musicgen_model.generate(**inputs, max_new_tokens=256)
    audio_tensor = audio_values[0].cpu()

    audio_numpy = audio_tensor.numpy()
    audio_numpy = audio_numpy / np.max(np.abs(audio_numpy))  # Normalisieren

    sr = musicgen_model.config.audio_encoder.sampling_rate or 32000

    # Temporäre WAV-Datei schreiben
    temp_path = tempfile.mktemp(suffix=".wav")
    sf.write(temp_path, audio_numpy.T, sr)  # Wichtig: Transponieren für mono/stereo

    return (sr, audio_numpy), temp_path
'''
def generate_music_with_download(prompt):
    inputs = processor(text=[prompt], return_tensors="pt").to("cuda")
    audio_values = musicgen_model.generate(**inputs, max_new_tokens=256)
    audio_tensor = audio_values[0].cpu()

    # Normalisieren
    audio_numpy = audio_tensor.numpy()
    audio_numpy = audio_numpy / np.max(np.abs(audio_numpy))  # [-1, 1]

    # Sampling-Rate
    sr = 32000  # Standard bei MusicGen small/medium/large

    # Temporäre WAV-Datei erzeugen
    temp_path = tempfile.mktemp(suffix=".wav")
    sf.write(temp_path, audio_numpy.T, sr)  # Achtung: .T nötig für shape [samples, channels]

    return (sr, audio_numpy), temp_path

# ----------- GRADIO UI -----------

with gr.Blocks() as demo:
    gr.Markdown("# Text-to-Image & Text-to-Music Generator")

    with gr.Row():
        with gr.Column():
            txt_prompt = gr.Textbox(label="Prompt")
            btn = gr.Button("Generate")

        with gr.Column():
            img_output = gr.Image(label="Generated Image")
            #audio_output = gr.Audio(label="Generated Music", type="numpy")
            music_output = gr.Audio(label="Generated Music", type="numpy")
            download_link = gr.File(label="Download WAV")

    btn.click(fn=generate_image, inputs=txt_prompt, outputs=img_output)
    #btn.click(fn=generate_music, inputs=txt_prompt, outputs=audio_output)


    btn.click(fn=generate_music_with_download,
              inputs=txt_prompt,
              outputs=[music_output, download_link])




demo.launch()