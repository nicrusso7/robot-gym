import os
import pathlib
import platform
from distutils.core import setup

from setuptools import find_packages

this_directory = pathlib.Path(__file__).parent.resolve()
os_name = platform.system()

with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read()


setup(
    name='robot_gym',
    version='0.0.1',
    license='Apache 2.0',
    packages=find_packages(),
    author='Nicola Russo',
    author_email='dott.nicolarusso@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/nicrusso7/robot-gym',
    download_url='https://github.com/nicrusso7/robot-gym/archive/master.zip',
    install_requires=requirements,
    entry_points='''
        [console_scripts]
        robot-gym=robot_gym.cli.entry_point:cli
    ''',
    include_package_data=True,
    keywords=['openai', 'gym', 'robotics', 'quadruped', 'pybullet', 'reinforcement learning', 'machine learning',
              'RL', 'ML', 'AI', 'tensorflow'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Framework :: Robot Framework :: Library',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7']
)
