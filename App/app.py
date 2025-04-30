import os
import uuid
import numpy as np
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline
from transformers import pipeline as hf_pipeline
import soundfile as sf

# -------------------------#
#   Modelldefinitionen     #
# -------------------------#

# Stable Diffusion-Pipeline laden (GPU, float16)
try:
    # TODO: test other diffusion models
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16,
        attn_implementation = "eager"
    )
    if torch.cuda.is_available():
        sd_pipe = sd_pipe.to("cuda")
        sd_pipe.enable_attention_slicing()
except Exception as e:
    print("Fehler beim Laden von Stable Diffusion:", e)
    sd_pipe = None

# MusicGen-Pipeline laden (text-to-audio)
try:
    # TODO: test medium and large for better results
    device = 0 if torch.cuda.is_available() else -1
    # TODO: tune the parameter
    mg_pipe = hf_pipeline("text-to-audio", model="facebook/musicgen-small", device=device)
except Exception as e:
    print("Fehler beim Laden von MusicGen:", e)
    mg_pipe = None


# -------------------------#
#   Generierungsfunktion   #
# -------------------------#

def generate(prompt: str):
    image = None
    audio_path = None

    # Bild mit Stable Diffusion generieren
    if sd_pipe is not None:
        try:
            ### TODO: Boosting the Prompt for better results prompt+text+text - stable specific
            result = sd_pipe(prompt)
            image = result.images[0]
        except Exception as e:
            print("Stable Diffusion-Fehler:", e)
            image = None

    # Musik mit MusicGen generieren
    if mg_pipe is not None:
        try:
            ### TODO: Boosting the Prompt for better results prompt+text+text - musicgen specific

            music_output = mg_pipe(prompt, generate_kwargs={"do_sample": True})


            ## Better Transformation
            # audio = music_output["audio"]  # NumPy-Array
            audio_tensor = music_output["audio"]
            if isinstance(audio_tensor, torch.Tensor):
                audio = audio_tensor.squeeze().cpu().numpy()
            else:
                audio = np.array(audio_tensor).squeeze()


            sr = music_output["sampling_rate"]  # Sample-Rate

            # Audio normalisieren und als int16 casten
            audio = np.array(audio, dtype=np.float32)
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            audio_int = (audio * 32767).astype(np.int16)

            # WAV-Datei speichern
            audio_filename = f"output_{uuid.uuid4().hex}.wav"
            sf.write(audio_filename, audio_int, sr)
            audio_path = audio_filename
        except Exception as e:
            print("MusicGen-Fehler:", e)
            audio_path = None

    return image, audio_path, audio_path


# -------------------------#
#   Gradio-Webanwendung    #
# -------------------------#

with gr.Blocks() as demo:
    gr.Markdown("# Text-zu-Bild und Musik-Generator")
    prompt_input = gr.Textbox(label="Prompt", placeholder="Beschreibung eingeben...", lines=2)
    output_image = gr.Image(label="Generiertes Bild")
    output_audio = gr.Audio(label="Generierte Musik", type="filepath")
    download_btn = gr.DownloadButton(label="WAV herunterladen")
    generate_btn = gr.Button("Generieren")

    generate_btn.click(fn=generate,
                       inputs=[prompt_input],
                       outputs=[output_image, output_audio, download_btn])

if __name__ == "__main__":
    demo.launch()