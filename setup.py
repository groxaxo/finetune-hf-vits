"""
Setup configuration for Soprano TTS with optional dependencies.
"""

from setuptools import setup, find_packages

# Read requirements
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Optional dependencies
extras_require = {
    "onnx": [
        "onnxruntime>=1.16.0",
        "onnx>=1.15.0",
    ],
    "openvino": [
        "openvino>=2024.0.0",  # OpenVINO 2025+ compatible
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
    ],
}

# Convenience: all optional dependencies
extras_require["all"] = list(set(sum(extras_require.values(), [])))

setup(
    name="soprano-tts",
    version="0.1.0",
    description="Soprano TTS with ONNX and OpenVINO CPU inference support",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Soprano TTS Contributors",
    packages=find_packages(include=["soprano", "soprano.*"]),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "soprano-export-decoder=soprano.export.decoder_export:main",
            "soprano-export-lm=soprano.export.lm_step_export:main",
            "soprano-bench=scripts.bench_cpu_rtf:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
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
