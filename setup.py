from setuptools import setup, find_packages

setup(
    name='dspg-utils',
    version='0.1.0',
    author='Sebastian Daza',
    author_email='sebastian.daza@teladochealth.com',
    description='Utils for the DSPG team',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Teladoc/dspg-utils',
    packages=find_packages(exclude=('tests',)),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=[
        # Add your project dependencies here. For example:
        'numpy>=1.21.5',
        'pandas>=1.4.4',
        'matplotlib>=3.5.2',
        'seaborn>=0.11.2',
		'matplotx>=0.3.10',
        'SharePlum>=0.5.1',
        'openpyxl>=3.1.2',
        'multiprocess>=0.70.14', 
        'statsmodels>=0.13.2', 
        'scipy>=1.9.1', 
        'psycopg2>=2.9.8'
    ],
)