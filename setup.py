import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aoc_helper",
    version="1.2.17",
    author="Starwort, salt-die",
    description="A helper package for Advent of Code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Starwort/aoc_helper",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["requests", "beautifulsoup4", "typing", "typing_extensions"],
    extras_require={"cli": ["click"], "colour": ["colorama"]},
)
