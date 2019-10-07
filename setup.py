import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='finrich',
    version='0.1.3',
    author='Anthony Aylward',
    author_email='aaylward@eng.ucsd.edu',
    description='Calculate enrichment of genomic regions with fine-mapping signals',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/anthony-aylward/finrich.git',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    install_requires=['pybedtools', ''],
    entry_points={
        'console_scripts': ['finrich=finrich.finrich:main']
    }
)
