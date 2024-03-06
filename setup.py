from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path:str)->List[str]:
    """
    this function reads the requirements file and returns a list of packages
    """
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements


setup(
    name='FraudDetection Using SEC 10-K Filings',
    version='0.0.1',
    description='Based on the MD&A section of the 10-K filings, this project aims to predict whether a company has likely committed fraud or not.',
    author='Haimonti Dutta',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)