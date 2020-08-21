from setuptools import find_packages, setup

setup(
    name="wutils",
    author="Matias Thayer",
    author_email="matias.thayer@gmail.com",
    url="https://github.com/chechir/wutils",
    use_scm_version=True,
    setup_requires=[
        "setuptools_scm",
    ],
    packages=find_packages(exclude=["tests"]),
    # entry_points={"console_scripts": ["test = wutils.main:cli"]},
    install_requires=(
        # 'click>=7',
        "numexpr",
        "pandas",
        "scikit-learn",
        "scipy",
        "numpy",
        # 'numpy>=1.19',
    ),
)
