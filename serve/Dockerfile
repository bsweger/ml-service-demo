FROM python:3.10-slim-bullseye

# Add a constraint to avoid installing cython 3.0
# This is a workaround to install PyYAML w/o error
# https://github.com/yaml/pyyaml/issues/736
RUN echo "cython<3" > /tmp/constraint.txt

COPY requirements.txt requirements.txt

RUN PIP_CONSTRAINT=/tmp/constraint.txt pip install -r requirements.txt

COPY src/ /src

WORKDIR /src

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]