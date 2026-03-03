.PHONY: install train quick eval-random replay test

install:
	python -m pip install -r requirements.txt

train:
	python train.py --timesteps 300000 --n-envs 8 --run-name ppo_pvz

quick:
	python train.py --quick --timesteps 50000 --n-envs 4 --run-name quick_test

eval-random:
	python eval.py --policy random --episodes 10

replay:
	python replay.py --model models/ppo_pvz_final.zip --policy ppo

test:
	pytest -q
