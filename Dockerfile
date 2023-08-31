FROM python:latest

WORKDIR /app

COPY requirements.txt .
COPY bot_script.py .
ADD fine-tuned_ccm2 ./fine-tuned_ccm2

RUN pip3 install -r requirements.txt

CMD ["python3", "bot_script.py"]
