from setuptools import setup, find_packages

setup(
    name="pyxations",
    version="0.1.0",
    author="NeuroLIAA",
    description="A Python package for handling eyetracking data",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NeuroLIAA/pyxations",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)