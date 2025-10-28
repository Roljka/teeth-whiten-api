#!/usr/bin/env bash
echo "Forcing Python 3.10 install..."
pyenv install -s 3.10.14
pyenv global 3.10.14
pip install -r requirements.txt
