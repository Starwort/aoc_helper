import os

from pypi_config import PASSWORD, USERNAME

os.system(f"twine upload -u {USERNAME} -p {PASSWORD} --skip-existing")
