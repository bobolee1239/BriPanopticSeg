from setuptools import setup, find_packages

setup(
    name='BriPanopticSeg',
    version='0.1',
    packages=find_packages(),  # Automatically finds Dataset/, Ut/, etc.
    install_requires=[
        'torch',
        'numpy',
        'albumentations',
        'matplotlib',
        'Pillow',
        # Add more if needed
    ],
    python_requires='>=3.8',
)

