import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="FICHnet",
    version="0.0.1",
    author="Yutong Chen",
    author_email="ychen146@mgh.harvard.edu",
    description="Predict long term outcomes after ICH",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Yutong441/FICHnet",
    packages=setuptools.find_namespace_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Creative Commons Noncommercial License",
        "Operating System :: Linux",
    ],
    python_requires='>=3.6',
    install_requires=[i.strip() for i in open("requirements.txt").readlines()],
)
