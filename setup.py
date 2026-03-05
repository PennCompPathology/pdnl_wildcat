from setuptools import setup, find_packages

setup(
    name='pdnl_wildcat',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'tqdm',
        'pdnl_sana'
    ],
    entry_points={
        "console_scripts":
        "pdnl_wildcat = pdnl_wildcat:main"
    },
)
