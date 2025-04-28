from setuptools import setup, find_packages

LATEST_VERSION = "0.0.10"

exclude_packages = []

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
with open("requirements.txt", "r") as f:
    reqs = [line.strip() for line in f if not any(pkg in line for pkg in exclude_packages)]

setup(
    name="deep-researcher",
    version=LATEST_VERSION,
    author="Jai Juneja",
    author_email="jai@qxlabs.com",
    description="A package for performing deep research using AI agents, implemented using the OpenAI Agents SDK",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qx-labs/agents-deep-research",
    package_dir={'deep_researcher': 'deep_researcher'},
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=reqs,
    extras_require={
        'dev': [
            'pytest',
            'pytest-asyncio'
        ]
    },
    entry_points={
        'console_scripts': [
            'deep-researcher=deep_researcher.main:cli_entry',
        ],
    },
)