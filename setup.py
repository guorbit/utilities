from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="guorbit_utils",
    version="0.1.0",
    author="GU Orbit Software Team",
    author_email="",
    description="A package containing utilities for GU Orbit Software",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(where="utilities"),
    package_dir={"": "utilities"},
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy >= 1.24.0",
        "rasterio >= 1.3.6",
        "Pillow >= 9.4.0",
        "tensorflow >= 2.10",
    ],
    extras_require={
        "dev": [
            "pytest >= 7.2.2",
            "pytest-cov >= 4.0.0",
            "twine >= 4.0.0",
        ]
    },
)