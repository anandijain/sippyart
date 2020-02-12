from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="sippysound",
    version="0.01",
    description="make music the nerdy way",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Anand Jain",
    author_email="anandj@uchicago.edu",
    packages=["sippysound"],  # same as name
    url="https://github.com/anandijain/audio",
    install_requires=[
        "numpy",
        "flask",
        "torch",
        "torchaudio",
        "torchvision",
    ],  # external packages as dependencies
    python_requires=">=3.6",
)
