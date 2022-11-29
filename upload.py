import os
import sys

from pypi_config import PASSWORD, USERNAME

os.system(f"{sys.executable} -m build -n")
os.system(f"twine upload dist/* -u {USERNAME} -p {PASSWORD} --skip-existing")
