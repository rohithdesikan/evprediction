# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.7.7-buster

LABEL author="Rohith Desikan <rohithdesikan@gmail.com>"

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Set workdir
WORKDIR /container

# Add necessary folders
COPY requirements.txt /container
COPY models /container
COPY app /container
COPY data/interim /container

# Install pip requirements
RUN pip install --upgrade pip
RUN pip --no-cache-dir install -r requirements.txt

# Expose a 5000 port
EXPOSE 5000

# Switching to a non-root user, please refer to https://aka.ms/vscode-docker-python-user-rights
# RUN useradd appuser && chown -R appuser /container
# USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
ENTRYPOINT [ "python" ]
CMD [ "app/app.py" ]