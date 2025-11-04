from setuptools import setup, find_packages

setup(
    name="demo-tables",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
    ],
    author="Sam Wakelin",
    description="Create demographics tables with summary statistics and statistical tests",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/demo-tables",
    python_requires=">=3.8",
)