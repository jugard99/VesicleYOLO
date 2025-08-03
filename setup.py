from setuptools import setup

setup(
   name='VesicleYOLO',
   version='1.0',
   description='A really really really rough way to use YOLO to quantify vesicles',
   author='Julian Gard',
   author_email='julianlgard@gmail.com',
   packages=['VesicleYOLO'],  #same as name
   install_requires=['wheel'], #external packages as dependencies
   scripts=[
            'executions/GUVfinal',
            'train/mouseTrain',
           ]
)