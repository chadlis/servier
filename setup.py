#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['pandas>=1.4.2',
'pandera>=0.10.1',
'scikit-learn>=1.0.2',
]

test_requirements = ['pytest>=3', ]

setup(
    author="Salah Chadli",
    author_email='salah.chadli@gmail.com',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Drug molecule properties prediction",
    entry_points={
        'console_scripts': [
            'servier=servier.cli:main',
        ],
    },
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='servier',
    name='servier',
    packages=find_packages(include=['servier', 'servier.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/chadlis/servier',
    version='0.1.0',
    zip_safe=False,
)
