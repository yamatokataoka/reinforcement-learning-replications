from setuptools import setup, find_packages

VERSION = '0.0.2'

long_description = """

# Quick reference

* **Github Repository**: [https://github.com/yamatokataoka/reinforcement-learning-replications](https://github.com/yamatokataoka/reinforcement-learning-replications)

# What is Reinforcement Learning Replications?

Reinforcement Learning Replications is a set of Pytorch implementations of reinforcement learning algorithms.

"""

setup(
  name='rl_replicas',
  version=VERSION,
  description='Reinforcement Learning Replications is a set of Pytorch implementations of reinforcement learning algorithms.',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author='Yamato Kataoka',
  url='https://github.com/yamatokataoka/reinforcement-learning-replications',
  packages=find_packages(),
  license='MIT',
  keywords='reinforcement-learning',
  install_requires=[
    'gym[atari, box2d, classic_control]',
    'numpy',
    'scipy',
    'torch>=1.4.0',
    'tensorboard'
  ],
  python_requires='>=3.6'
)
