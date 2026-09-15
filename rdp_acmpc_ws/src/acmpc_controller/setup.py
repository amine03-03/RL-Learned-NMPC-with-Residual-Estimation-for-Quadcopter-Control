from setuptools import setup

package_name = "acmpc_controller"

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
    description="The ACMPC control node, PX4 frame conversions, the B_d bridge and the §9.4 preflight checks.",
    license="MIT",
    entry_points={
        "console_scripts": [
            'check_ctbr = acmpc_controller.check_ctbr:main',
            'check_glue = acmpc_controller.check_glue:main',
            'controller_node = acmpc_controller.controller_node:main',
        ],
    },
)
