.PHONY: setup run clean docker-build docker-run


setup:
		pip3 install -r requirements.txt


run:
		python3 main.py

clean:
		rm -rf logs/ outputs/ __pycache__src/__pycache__

docker-build:
		docker build -t wind-turbine-maintainer .

docker-run:
		docker run --rm -v $(PWD)/outputs:/app/outputs -v $(PWD)/logs:/app/logs wind-turbine-maintainer