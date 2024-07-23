from setuptools import setup, find_packages


version = {}
with open("./neutral_atom_image_analysis/version.py") as f:
    exec(f.read(), version)
__version__ = version["__version__"]


def readme():
    with open('README.md') as f:
        return f.read()

setup(name='neutral_atom_image_analysis',
      version=__version__,
      description='Package containing image analysis algorithms for neutral atom quantum computers',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
      ],
      keywords='data analysis',
      url='',
      author='Jonas Winklmann',
      author_email='jonas.winklmann@tum.de',
      license_files='LICENSE',
      packages=find_packages(),
      entry_points={
            'console_scripts': [
                  # 'script_name = package.module:function'
            ],
            'gui_scripts': []
      },
      install_requires=[
            'numpy',
      ],
      python_requires='>=3.4',
      include_package_data=True,
      package_data={'': ['*.so','*.dll','*.pyd']},
      zip_safe=False)