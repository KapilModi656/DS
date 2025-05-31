from setuptools import setup, find_packages
def get_requirements(file_path):
    """
    This function reads a requirements file and returns a list of packages.
    It removes any comments or empty lines.
    """
    with open(file_path, 'r') as file:
        requirements = file.readlines()
    # Remove comments and empty lines
    requirements = [line for line in requirements if not line.startswith('#') and line.strip()]
    
    # Clean up the requirements list
    requirements = [line.strip() for line in requirements if line.strip() and not line.startswith('#')]

    # Remove any editable installs
    requirements = [req for req in requirements if not req.startswith('-e')]
    
    return requirements
setup(
    name='Project1',
    version='0.1',
    description='A sample project for demonstration purposes',
    author='Kapil Modi',
    author_email='kapilmodi.656@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
