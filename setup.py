from setuptools import setup, find_packages



setup(
    name="feateml",
    author="khdoex,mrp3anut,kutayisgorur",
   
    
    url="https://github.com/mrp3anut/featureml",
    
    packages=find_packages(),
    install_requires=[
   'numpy',
   'scipy',
   'pywt',
   'obspy',
	 'jupyter'], 

    python_requires='>=3.6',
)

