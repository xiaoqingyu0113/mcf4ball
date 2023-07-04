from setuptools import setup, find_packages

setup(
    name='mcf4ball',
    version='0.1.0',
    description='multi-camera fusion for estimation of 3D ball trajectory',
    author='Qingyu Xiao',
    author_email='xiaoqingyu0113@gmail.com',
    url='https://github.com/xiaoqingyu0113/mcf4ball',
    packages=find_packages(include=['mcf4ball', 'mcf4ball.*']),
    install_requires=[
        'PyYAML',
        'numpy'
    ],
    extras_require={'plotting': ['matplotlib']},
)