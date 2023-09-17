FROM python:3.10-slim

RUN apt-get update && apt install curl -y
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

EXPOSE ${BACKEND_PORT}

COPY ./start.bash /code/start.bash
RUN chmod +x /code/start.bash

ENTRYPOINT ["./start.bash"]