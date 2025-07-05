from setuptools import setup, find_packages

setup(
    name="dental-ai-bridge",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "python-multipart>=0.0.6",
        "pillow>=10.0.1",
        "numpy>=1.24.3",
        "torch>=2.1.0",
        "tensorflow>=2.13.0",
    ],
)