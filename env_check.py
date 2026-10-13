import sys
import subprocess
try:
    import pkg_resources
except ImportError:
    pass
print('Python:', sys.version)
packages = ['scikit-survival', 'lifelines', 'rulekit', 'GEOparse']
for p in packages:
    try:
        import importlib
        importlib.import_module(p.replace('-', '_'))
        print(f'{p}: Installed')
    except ImportError:
        print(f'{p}: Not installed')

print('\nChecking Java:')
subprocess.run(['java', '-version'], shell=True)

print('\nChecking R:')
subprocess.run(['R', '--version'], shell=True)
