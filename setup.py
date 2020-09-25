try:
    from setuptools import setup

    kw = {"test_suite": "tests"}
except ImportError:
    from distutils.core import setup

    kw = {}

setup(
    name="penty",
    version="0.0.1",
    packages=["penty"],
    description="Type numpy kernels",
    long_description="""
Type checking for Numpy kernels, and more

Given an entry point, penty verifies that the execution following that entry
point is correctly typed. It can use that information to generate different,
non-polymorphic, overloads of these functions.""",
    author="serge-sans-paille",
    author_email="serge.guelton@telecom-bretagne.eu",
    url="https://github.com/serge-sans-paille/penty/",
    license="BSD 3-Clause",
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Development Status :: 5 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    python_requires=">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*",
    **kw
)
