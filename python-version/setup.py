from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('../LICENSE') as f:
    license = f.read()

setup(
    name='RRTNCTTASA',
    version='0.0.3',
    description='Robust Real-Time Non-Cooperative Target Tracking Algorithm for Space Applications',
    long_description=readme,
    author='Mikkel Anderson & Jonas Nielsen',
    author_email='s184230@dtu.dk',
    url='https://github.com/mikk5829/RRTNCTTASA',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)