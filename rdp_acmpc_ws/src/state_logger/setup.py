from setuptools import setup

package_name = "state_logger"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="Amine Bouzid",
    maintainer_email="bouzid.amine1010@gmail.com",
    description="50 Hz synchronised CSV writer.",
    license="MIT",
    entry_points={
        "console_scripts": [
            'logger_node = state_logger.logger_node:main',
        ],
    },
)
