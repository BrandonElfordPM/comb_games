from setuptools import setup

setup(name='DiskonnectPlayer',
      version='0.0',
	author='Brandon Elford',
	description='Gym environment for 1D-Diskonnect player, a combinatorial scoring game studied during my honours research in undergrad.',
      install_requires=['gym','numpy','torch','wandb','stable-baselines3', 'Cython']
)
