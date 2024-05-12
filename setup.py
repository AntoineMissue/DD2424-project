"""
Defines package metadata and allows for installation.
"""
import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"

SRC_REPO = "nlpProject"
GITHUB_URL = "https://github.com/AntoineMissue/DD2424-project"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author="Group 14",
    description="A deep learning project to learn NLP methods and LSTM neural networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=GITHUB_URL,
    project_urls={
        "Bug Tracker": f"{GITHUB_URL}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
