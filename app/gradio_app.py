"""Gradio application for running Tortoise TTS interactively."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch

from tortoise.api import MODELS_DIR, TextToSpeech
from tortoise.utils.audio import get_voices, load_voices


DEFAULT_TEXT = "Tortoise makes it easy to generate natural sounding speech."
PRESETS = ["ultra_fast", "fast", "standard", "high_quality"]
SAMPLE_RATE = 24000


def _resolve_device() -> str:
    """Return the device that should be used for inference."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _load_tts() -> TextToSpeech:
    """Load and cache a ``TextToSpeech`` instance."""
    device_override = os.environ.get("TORTOISE_GRADIO_DEVICE")
    device = device_override or _resolve_device()
    autoregressive_batch_size: Optional[int]
    if device == "cpu":
        # CPU inference benefits from a smaller batch size to avoid excessive memory usage.
        autoregressive_batch_size = 1
    else:
        autoregressive_batch_size = None
    return TextToSpeech(
        models_dir=MODELS_DIR,
        use_deepspeed=False,
        autoregressive_batch_size=autoregressive_batch_size,
        device=device,
    )


@lru_cache(maxsize=None)
def _load_voice_latents(voice: str) -> Tuple[Optional[List[torch.Tensor]], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """Load reference audio/latents for a specific voice."""
    clips, conditioning = load_voices([voice])
    return clips, conditioning


@lru_cache(maxsize=1)
def _available_voices() -> List[str]:
    """Return the list of bundled voices plus the ``random`` voice."""
    voices: Dict[str, List[str]] = get_voices()
    choices = sorted(voices.keys())
    return ["random", *choices]


def synthesize_speech(text: str, voice: str, preset: str, seed: Optional[int] = None) -> Tuple[int, np.ndarray]:
    """Generate speech for the provided text and voice selection."""
    if not text or not text.strip():
        raise gr.Error("Please provide some text to synthesize.")
    if voice not in _available_voices():
        raise gr.Error("The requested voice is not available.")

    voice_samples = None
    conditioning_latents = None
    if voice != "random":
        voice_samples, conditioning_latents = _load_voice_latents(voice)

    tts = _load_tts()
    audio = tts.tts_with_preset(
        text,
        preset=preset,
        voice_samples=voice_samples,
        conditioning_latents=conditioning_latents,
        use_deterministic_seed=seed,
    )

    if isinstance(audio, list):
        # ``tts_with_preset`` returns either a tensor or a list of tensors depending on ``k``.
        audio = audio[0]
    waveform = audio.squeeze().cpu().numpy()
    return SAMPLE_RATE, waveform


def build_interface() -> gr.Blocks:
    """Construct and return the Gradio Blocks interface."""
    voices = _available_voices()

    with gr.Blocks(title="Tortoise TTS") as demo:
        gr.Markdown(
            """
            # Tortoise TTS

            Generate speech with the [Tortoise](https://github.com/neonbjb/tortoise-tts) models.
            Downloading the model weights may take a few minutes the first time you run the app.
            """
        )

        with gr.Row():
            text_input = gr.Textbox(
                label="Text",
                value=DEFAULT_TEXT,
                lines=4,
                placeholder="Enter the text you would like to synthesize...",
            )
        with gr.Row():
            voice_input = gr.Dropdown(voices, value=voices[0], label="Voice")
            preset_input = gr.Dropdown(PRESETS, value="fast", label="Quality preset")
            seed_input = gr.Number(
                label="Deterministic seed",
                value=None,
                precision=0,
                placeholder="Optional",
            )
        generate_button = gr.Button("Generate speech")
        audio_output = gr.Audio(label="Generated audio", type="numpy")

        generate_button.click(
            synthesize_speech,
            inputs=[text_input, voice_input, preset_input, seed_input],
            outputs=audio_output,
        )

        gr.Examples(
            examples=[
                ["The future belongs to those who believe in the beauty of their dreams.", "random", "fast", None],
                ["Hello there! This is the Tortoise speech system speaking.", voices[min(1, len(voices) - 1)], "ultra_fast", None],
            ],
            inputs=[text_input, voice_input, preset_input, seed_input],
            label="Try these sample prompts:",
        )

        gr.Markdown(
            """
            ## Tips
            * Generating audio can take a while, especially on CPU-only machines.
            * Provide a seed value if you want reproducible results.
            * Select `random` to sample a new synthetic voice for each run.
            """
        )

    return demo


def main() -> None:
    """Entry point used when executing the module as a script."""
    demo = build_interface()
    demo.queue().launch()


if __name__ == "__main__":  # pragma: no cover - manual invocation hook
    main()
