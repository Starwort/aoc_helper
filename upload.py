import os

from pypi_config import PASSWORD, USERNAME

os.system("python setup.py sdist")
os.system(f"twine upload dist/* -u {USERNAME} -p {PASSWORD} --skip-existing")
