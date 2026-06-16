from setuptools import setup, find_packages

setup(
    name="mil",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch==2.7.1",
        "torchvision==0.22.1",
        "pandas",
        "scikit-learn",
        "transformers==4.57.1",
        "tqdm",
        "Pillow",
        "openslide-python",
        "timm",
    ],
)

