from importlib.metadata import version, PackageNotFoundError
import pandas as pd
from tabulate import tabulate

def check_package_versions():
    # Required packages from requirements.txt
    required_packages = [
        'Flask',
        'Flask-SQLAlchemy',
        'Flask-Login',
        'Werkzeug',
        'joblib',
        'pandas',
        'numpy',
        'xgboost',
        'scikit-learn',
        'requests',
        'python-dateutil',
        'SQLAlchemy',
        'Jinja2',
        'MarkupSafe',
        'itsdangerous',
        'click',
        'typing-extensions',
        'six',
        'urllib3',
        'certifi',
        'idna',
        'setuptools',
        'wheel',
        'greenlet',
        'blinker',
        'tabulate'
    ]
    
    # Create lists for package names and versions
    packages = []
    versions = []
    
    # Check each required package
    for package_name in required_packages:
        try:
            package_version = version(package_name)
            packages.append(package_name)
            versions.append(package_version)
        except PackageNotFoundError:
            packages.append(package_name)
            versions.append('Not installed')
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Package': packages,
        'Installed Version': versions
    })
    
    # Sort alphabetically by package name
    df = df.sort_values('Package')
    
    # Print the table
    print("\nCurrent Package Versions:")
    print("========================")
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    
    # Check for any not installed packages
    not_installed = df[df['Installed Version'] == 'Not installed']
    if not not_installed.empty:
        print("\nWarning: The following packages are not installed:")
        for package in not_installed['Package']:
            print(f"- {package}")
        print("\nTo install missing packages, run:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    check_package_versions()