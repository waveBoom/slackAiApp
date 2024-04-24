FROM python:3.9-slim

ENV PORT=8080

USER root

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir /data && mkdir /data/myGPTReader

RUN apt-get update && apt-get install -y wget

RUN wget http://security.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.22_amd64.deb \
    && dpkg -i libssl1.1_1.1.1f-1ubuntu2.22_amd64.deb \
    && wget http://security.ubuntu.com/ubuntu/pool/main/o/openssl/libssl-dev_1.1.1f-1ubuntu2.22_amd64.deb \
    && dpkg -i libssl-dev_1.1.1f-1ubuntu2.22_amd64.deb

RUN apt-get clean && rm -rf /var/lib/apt/lists/* \
    && rm *.deb

EXPOSE 8080

CMD ["gunicorn", "app.server:app"]