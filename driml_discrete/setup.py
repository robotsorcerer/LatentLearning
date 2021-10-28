import setuptools

INSTALL_REQUIRES = [
    'numpy',
    'torch',
]
TEST_REQUIRES = [
    # testing and coverage
    'pytest', 'coverage', 'pytest-cov',
    # unmandatory dependencies of the package itself
    'atari_py', 'opencv-python', 'psutil', 'pyprind', 'gym',
]

setuptools.setup(
    name='noisy_state_abstractions',
    version='0.0.1',
    packages=setuptools.find_packages(),
    license='MIT License',
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'test': TEST_REQUIRES + INSTALL_REQUIRES,
    },

)