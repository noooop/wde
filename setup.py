from setuptools import setup

setup(
    name="wde",
    version="v0.2.2",
    author="noooop",
    author_email="noooop@live.cn",
    description="Workflow Defined Engine",
    url="https://github.com/noooop/wde",
    package_dir={"wde": "wde"},
    entry_points={
        "console_scripts": [
            "wde=wde.cli:main",
        ],
    },
)