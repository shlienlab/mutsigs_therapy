from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name="mutsigs_therapy",
    version="0.1.0",
    description="A package to reproduce figures in the manuscript",
    author="Mehdi Layeghifard",
    author_email="mlayeghi@gmail.com",
    url="https://github.com/shlienlab/mutsigs_therapy",
    packages=find_packages(where="scripts"),  # Include scripts folder if needed
    install_requires=required_packages,  # Install dependencies
    python_requires=">=3.11.6",
    include_package_data=True,  # Ensure non-Python files (like notebooks) are included in source distribution
    entry_points={
        "console_scripts": [
            "my_script=scripts.my_script:main",  # Example CLI command
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

