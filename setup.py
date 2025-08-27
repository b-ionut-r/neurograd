from setuptools import setup, find_packages

# Read README.md for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neurograd",
    version="0.8.7",
    author="Bujor Ionut Raul",
    author_email="b-ionut-r@users.noreply.github.com",
    description="A Pure Python Deep Learning Framework with Automatic Differentiation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/b-ionut-r/neurograd",
    project_urls={
        "Bug Tracker": "https://github.com/b-ionut-r/neurograd/issues",
        "Documentation": "https://github.com/b-ionut-r/neurograd#readme",
        "Repository": "https://github.com/b-ionut-r/neurograd",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education", 
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "gpu": ["cupy-cuda12x>=13.5.1"],
        "visualization": ["matplotlib>=3.3.0"],
        "examples": [
            "scikit-learn>=0.24.0",
            "matplotlib>=3.3.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "matplotlib>=3.3.0",
            "scikit-learn>=0.24.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "all": [
            "cupy-cuda12x>=12.0.0",
            "matplotlib>=3.3.0",
            "scikit-learn>=0.24.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "pytest>=6.0.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
    },
    keywords=[
        "deep-learning",
        "neural-networks", 
        "automatic-differentiation",
        "machine-learning",
        "pytorch-like",
        "python",
        "gpu",
        "cuda",
        "conv2d",
        "backpropagation"
    ],
)
