.PHONY: venv install run clean

PY := python3
PIP := pip
VENV := .venv

venv:
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate; $(PIP) install --upgrade pip

install: venv
	. $(VENV)/bin/activate; $(PIP) install -r requirements.txt

run:
	. $(VENV)/bin/activate; $(PY) main.py

clean:
	rm -rf $(VENV) __pycache__ */__pycache__

