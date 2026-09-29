install:
	pip install -r requirements.txt

augment:
	python augment_dataset.py

train-face:
	python train_face.py

train-eyes-regions:
	python train_eyes_widerface.py

train-eyes:
	python train_eyes.py

train-mouth:
	python train_mouth.py

all: augment train-face train-eyes-regions train-eyes train-mouth

# Legacy — earlier two-model design, kept for reference. See README.
train-states:
	python train_states.py

validate:
	python validate.py
