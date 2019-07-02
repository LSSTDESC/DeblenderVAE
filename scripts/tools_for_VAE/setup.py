import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tools_for_VAE",
    version="0.0.1",
    author="Bastien Arcelin",
    author_email="arcelin@apc.in2p3.fr",
    description="test package for Deblender/VAE project",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages()
)
