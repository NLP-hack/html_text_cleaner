FROM python:3.10-slim

RUN apt-get update && apt install curl -y

WORKDIR /code

COPY ./model_files /code/model_files
RUN pip install gdown && \
    gdown --id 1PLLIyqfMz0MVc3CvrC54bVoR1z7waRt6 --output /code/model_files/bert_punct_spelling.pt

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

EXPOSE ${BACKEND_PORT}

COPY ./start.bash /code/start.bash
RUN chmod +x /code/start.bash

ENTRYPOINT ["./start.bash"]