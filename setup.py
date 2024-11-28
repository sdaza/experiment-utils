from setuptools import setup, find_packages

setup(
    name='experiment-utils',
    version='0.0.1',
    author='Sebastian Daza',
    author_email='sebastian.daza@gmail.com',
    description='Utils for experimental design and analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/sdaza/experiment-utils.git',
    packages=find_packages(exclude=('tests',)),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=[
        'numpy>=1.21.5',
        'pandas>=1.4.4',
        'matplotlib>=3.5.2',
        'seaborn>=0.11.2',
        'multiprocess>=0.70.14', 
        'statsmodels>=0.13.2', 
        'scipy>=1.9.1', 
		'dowhy==0.11.1',
		'linearmodels==6.1'
    ],
)