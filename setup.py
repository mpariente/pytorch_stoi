from setuptools import setup
from setuptools import find_packages

with open("README.md", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='torch_stoi',
    version='0.0.1',
    description='Computes Short Term Objective Intelligibility in PyTorch',
    author='Manuel Pariente',
    author_email='pariente.mnl@gmail.com',
    url='https://github.com/mpariente/torch_stoi',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    install_requires=['numpy', 'torch', 'pystoi'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    packages=find_packages()
)
