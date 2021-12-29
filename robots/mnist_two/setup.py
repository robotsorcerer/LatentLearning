from setuptools import setup

package_name = 'mnist_two'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'scipy', 'gym'],
    zip_safe=True,
    maintainer='Alex',
    maintainer_email='LambAlex@microsoft.com',
    description='TODO: Package description',
    license='Microsoft License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mnist_states = mnist_two.main:main'
        ],
    },
)
