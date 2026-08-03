rm -rf ./dist
python setup.py bdist_wheel
rm -rf *.egg-info
pip uninstall msadapter -y && pip install dist/*.whl