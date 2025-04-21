FROM python:3.11

WORKDIR /usr/src/app

COPY . .

# プロジェクトをインストール
RUN pip install --no-cache-dir -e .

CMD [ "python", "./main.py" ]
