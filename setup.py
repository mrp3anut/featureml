from setuptools import setup, find_packages



setup(
    name="featureml",
    author="khdoex,mrp3anut,EnesKutay",
   
    
    url="https://github.com/boun-earth-ml/earth-ml/featureml",
    
    packages=find_packages(),
    install_requires=[
   'numpy',
   'scipy',
   'PyWavelets',
   'obspy',
   'jupyter'], 

   )

