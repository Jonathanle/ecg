
FROM python:3.9  

WORKDIR /app
COPY . .

# Add your dependencies
RUN pip install -r requirements.txt

CMD ["python", "app.py"]