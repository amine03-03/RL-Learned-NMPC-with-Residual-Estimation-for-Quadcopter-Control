from setuptools import setup

package_name = "visualization"

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
    description="Live panel, the real-time Z-X demo panel and the offline figures R-F1..R-F13.",
    license="MIT",
    entry_points={
        "console_scripts": [
            'live_panel = visualization.panels:main',
            'live_zx = visualization.live_zx:main',
        ],
    },
)
