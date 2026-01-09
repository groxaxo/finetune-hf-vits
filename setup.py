"""
Setup configuration for VITS/MMS fine-tuning utilities.
"""

from setuptools import setup, find_packages

# Read requirements
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="finetune-hf-vits",
    version="1.0.0",
    description="Fine-tune VITS and MMS text-to-speech models using HuggingFace tools",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="VITS/MMS Finetuning Contributors",
    packages=find_packages(include=["utils", "utils.*", "monotonic_align", "monotonic_align.*", "enhancements", "enhancements.*"]),
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
)
