"""
ConjLean package setup.

Installs the ``conjlean`` Python package from the ``src/`` directory.
Requires Python 3.9+ for asyncio, walrus operator, and type hint support.
"""

from setuptools import find_packages, setup

setup(
    name="conjlean",
    version="0.1.0",
    description=(
        "ConjLean/REFUTE: automated mathematical conjecture generation, "
        "Lean 4 formal verification, and multi-agent counterexample search. "
        "ICML AI4Research 2026."
    ),
    author="ConjLean Authors",
    python_requires=">=3.9",
    package_dir={"conjlean": "src"},
    packages=["conjlean"],
    install_requires=[
        "anthropic>=0.40.0",
        "openai>=1.50.0",
        "google-generativeai>=0.8.0",
        "huggingface-hub>=0.25.0",
        "transformers>=4.45.0",
        "torch>=2.4.0",
        "sympy>=1.13.0",
        "numpy>=1.26.0",
        "pydantic>=2.9.0",
        "pyyaml>=6.0.2",
        "tqdm>=4.67.0",
        "rich>=13.9.0",
        "aiohttp>=3.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.0",
            "pytest-asyncio>=0.24.0",
            "pytest-mock>=3.14.0",
            "ruff>=0.9.0",
        ],
        "local_hf": [
            "peft>=0.13.0",
            "trl>=0.12.0",
            "bitsandbytes>=0.44.0",
            "datasets>=3.0.0",
            "accelerate>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "conjlean=run:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
