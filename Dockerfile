FROM python:3.9
COPY . /app
WORKDIR /app
COPY /app/.github/workflows/demo.py /app
RUN pip install -r requirements.txt
RUN python demo.py
EXPOSE $PORT
CMD gunicorn --workers=1 --bind 0.0.0.0:$PORT app:app
