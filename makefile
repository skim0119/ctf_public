PYVERSION?=python

test:
	${PYVERSION} run.py --epoch 2000

train:
	${PYVERSION} run.py --epoch 50000
