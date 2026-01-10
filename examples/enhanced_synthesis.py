#!/usr/bin/env python3
"""
Example: Using enhanced VITS/MMS with quality improvements.
"""

import argparse
import scipy.io.wavfile
from pathlib import Path
from transformers import pipeline

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from enhancements.postprocessing import enhance_tts_output


def main():
    parser = argparse.ArgumentParser(description="Enhanced VITS/MMS synthesis")
    parser.add_argument("--model", type=str, default="ylacombe/vits_ljs_welsh_female_monospeaker_2")
    parser.add_argument("--text", type=str, default="Hello, this is enhanced synthesis.")
    parser.add_argument("--preset", type=str, default="balanced", choices=["minimal", "balanced", "maximum"])
    parser.add_argument("--output", type=str, default="./output")
    args = parser.parse_args()
    
    print(f"Synthesizing with preset: {args.preset}")
    synthesiser = pipeline("text-to-speech", model=args.model, device=-1)
    speech = synthesiser(args.text, noise_scale=0.667)
    
    enhanced = enhance_tts_output(speech["audio"][0], speech["sampling_rate"], args.preset)
    
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)
    output_file = output_path / f"enhanced_{args.preset}.wav"
    
    scipy.io.wavfile.write(output_file, speech["sampling_rate"], enhanced)
    print(f"✓ Saved: {output_file}")


if __name__ == "__main__":
    main()
