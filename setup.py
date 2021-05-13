from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(name='Pychastic',
      version='0.1.2',
      description='Solvers for stochastic differential equations',
      url='https://github.com/RadostW/stochastic',
      author='Radost Waszkiewicz & Maciej Bartczak',
      author_email='radost.waszkiewicz@gmail.com',
      long_description=long_description,
      long_description_content_type='text/markdown',  # This is important!
      license='MIT',
      packages=['paczka'],
      zip_safe=False)
