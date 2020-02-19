from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="sippyart",
    version="0.01",
    description="make music the nerdy way",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Anand Jain",
    author_email="anandj@uchicago.edu",
    packages=["sippyart"],  # same as name
    url="https://github.com/anandijain/audio",
    install_requires=[
        "numpy",
        "flask",
        "torch",
        "torchaudio",
        "torchvision",
        "tensorboard",
    ],  # external packages as dependencies
    python_requires=">=3.6",
)
