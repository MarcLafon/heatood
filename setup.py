import setuptools

with open('requirements.txt') as open_file:
    install_requires = open_file.read()

setuptools.setup(
    name='heat',
    version='1.0.0',
    packages=[''],
    url='https://github.com/MarcLafon/heatood',
    license='MIT',
    author='Marc Lafon',
    author_email='marc.lafon@lecnam.net',
    description='Official code for HEAT.',
    python_requires='>=3.6',
    install_requires=install_requires
)
