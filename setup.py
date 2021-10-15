from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(name='pychastic',
      version='0.1.9',
      description='Solvers for stochastic differential equations',
      url='https://github.com/RadostW/stochastic',
      author='Radost Waszkiewicz & Maciej Bartczak',
      author_email='radost.waszkiewicz@gmail.com',
      long_description=long_description,
      long_description_content_type='text/markdown',  # This is important!
      project_urls = {
          'Documentation': 'https://pychastic.readthedocs.io',
          'Source': 'https://github.com/RadostW/stochastic'
      },
      license='MIT',
      install_requires = ['jax','jaxlib','numpy','sortedcontainers'],
      packages=['pychastic'],
      zip_safe=False)
