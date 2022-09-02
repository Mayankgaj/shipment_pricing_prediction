from setuptools import setup, find_packages
from typing import List

# Declaring variables for setup functions
PROJECT_NAME = "shipment-pricing-prediction"
VERSION = "0.0.1"
AUTHOR = "Mayank Gajbhiye"
DESCRIPTION = """The market for supply chain analytics is expected to develop at a CAGR of 17.3 percent
                 from 2019 to 2024, more than doubling in size. This data demonstrates how supply
                 chain organizations are understanding the advantages of being able to predict what will
                 happen in the future with a decent degree of certainty. Supply chain leaders may use
                 this data to address supply chain difficulties, cut costs, and enhance service levels all at
                 the same time."""
REQUIREMENT_FILE_NAME = "requirements.txt"
HYPHEN_E_DOT = "-e ."


def get_requirements_list() -> List[str]:
    """
    Description: This function is going to return list of requirement
    mention in requirements.txt file
    return This function is going to return a list which contain name
    of libraries mentioned in requirements.txt file
    """
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
        requirement_list = [requirement_name.replace("\n", "") for requirement_name in requirement_list]
        if HYPHEN_E_DOT in requirement_list:
            requirement_list.remove(HYPHEN_E_DOT)
        return requirement_list


setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=get_requirements_list()
)
