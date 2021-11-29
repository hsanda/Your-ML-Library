import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nyml",
    version="1.0.0",
    author="Bryan Gass",
    author_email="bagass@wpi.edu",
    description="Not-Your-ML is a library made for the process of learning what it means to build a library and to further explore the depths of deep learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    project_urls={
        "Documentation": "https://nyml.readthedocs.io/en/stable",
        "Source Code"  : "https://github.com/BeeGass/The-Learning-Library",
    },
    install_requires=[
        "numpy",
        "pre-commit==2.13.0",
        "torch",
        "flax",
        "tqdm",
    ],
    extras_require={
        "jax": [
            "jax>=0.1.71",
            "jaxlib>=0.1.49",
        ]
    },
    requires_python=">=3.8",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education/Academia",
        "Framework :: Sphinx",
        "Framework :: Pytest",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Deep Learning",
        "Topic :: Scientific/Engineering :: Supervised Learning",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)