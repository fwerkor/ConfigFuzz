from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="msadapter",
    version="0.6.0",
    author="msadapter",
    author_email="pcl.openi@pcl.ac.cn",
    description="msadaptor project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitee.com/mindspore/msadapter",
    project_urls={
        "Bug Tracker": "https://gitee.com/mindspore/msadapter/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
    ],
    packages=find_packages(include=["msadapter", "msa_thirdparty"]),
    package_dir={
        "msadapter": "msadapter",
        "msa_thirdparty": "msa_thirdparty",
    },
    package_data={
        'msadapter': ['*', '*/*', '*/*/*', '*/*/*/*', '*/*/*/*/*', '*/*/*/*/*/*'],
        'msa_thirdparty': ['*', '*/*', '*/*/*', '*/*/*/*', '*/*/*/*/*', '*/*/*/*/*/*']
    },
    python_requires=">=3.9",
    install_requires=[
        "mindspore>=2.7.1",
        "requests",
        "ml_dtypes",
    ],
)
