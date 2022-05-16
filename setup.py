from setuptools import setup, find_packages

setup(name="recsyskit",
      version="1.0",
      author="lzcpapa",
      author_email="miningsecret@gmail.com",
      description="",
      python_requires='>=3.6',
      packages=[
          package for package in find_packages()
          if package.startswith('recsyskit')
      ])
