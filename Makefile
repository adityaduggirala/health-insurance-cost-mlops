.PHONY: install train serve test docker
install:
	pip install -r requirements.txt && pip install -e .
train:
	python -m insurance.train
serve:
	uvicorn insurance.api:app --reload
test:
	pytest -q
docker:
	docker build -t insurance-api . && docker run --rm -p 8000:8000 insurance-api
