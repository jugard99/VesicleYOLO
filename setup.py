from setuptools import setup,find_packages

setup(
   name='VesicleYOLO',
   version='1.0',
   description='A really really really rough way to use YOLO to quantify vesicles',
   author='Julian Gard',
   author_email='julianlgard@gmail.com',
   packages=find_packages(),  #same as name
   install_requires=['wheel'], #external packages as dependencies
   scripts=[
            'VesicleYOLO/execution/GUVfinal.py',
            'VesicleYOLO/train/mouseTrain.py',
           ]
)