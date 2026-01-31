from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as f:
        requirements = [
            req.strip()
            for req in f.readlines()
            if req.strip() and req.strip() != HYPHEN_E_DOT
        ]
    return requirements

setup(
    name="machine-learning-project",
    version="0.0.1",
    author="Megha Ukkali",
    author_email="meghaukkali99@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
