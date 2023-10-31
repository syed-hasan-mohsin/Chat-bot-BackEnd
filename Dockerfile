FROM python:3.10
ENV PYTHONUNBUFFERED 1
RUN python -m venv /opt/venv
RUN mkdir /code
RUN mkdir /var/log/uwsgi
RUN touch /var/log/uwsgi/dashboard.log
WORKDIR /code
ENV PATH="/opt/venv/bin:$PATH"
ADD requirements.txt /code/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ADD . /code/