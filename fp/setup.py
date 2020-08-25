from setuptools import setup

setup(name='CREAM',
      version='1.0.4',
      url='https://github.com/czodrowskilab/gafp',
      author='Marcel Baltruschat',
      packages=['ctrainlib'],
      python_requires='>=3.6',
      license='MIT',
      entry_points={
          'console_scripts': [
              'cream = ctrainlib.__main__:main',
          ],
      }
      )
