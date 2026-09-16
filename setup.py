# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="trdg",
    version="2.0.0",
    description="ThaiTextRecognitionDataGenerator: synthetic Thai/English line images for OCR training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Belval/TextRecognitionDataGenerator",
    author="Edouard Belval",
    author_email="edouard@belval.org",
    # Choose your license
    license="MIT",
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="synthetic data text-recognition training-set-generator ocr dataset fake text",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    include_package_data=True,
    install_requires=[
        "uharfbuzz>=0.39",
        "fonttools>=4.40",
        "pillow>=10.0",
        "numpy>=1.24",
        "opencv-python>=4.8",
        "tqdm>=4.60",
        "requests>=2.28",
    ],
    extras_require={
        "wiki": ["wikipedia>=1.4.0", "pyarrow"],
        "arabic": ["arabic-reshaper", "python-bidi"],
    },
    entry_points={
        "console_scripts": [
            "trdg=trdg.run:main",
            "trdg-corpus=trdg.corpus:main",
        ],
    },
)
