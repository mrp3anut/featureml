from setuptools import setup, find_packages



setup(
    name="featureml",
    author="khdoex,mrp3anut,EnesKutay",
   
    
    url="https://github.com/mrp3anut/featureml",
    
    packages=find_packages(),
    setup_requires=['numpy'],
    install_requires=[
   'scipy',
   'PyWavelets',
   'obspy'], 

   )

