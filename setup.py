from setuptools import setup, find_packages

setup(
    name="procedural-gpt",
    version="0.1.0",
    description="Procedural GPT: Guaranteeing Generative Validity via Executable Latent States",
    author="Anonymous",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "typing-extensions>=4.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
        ],
    },
)
