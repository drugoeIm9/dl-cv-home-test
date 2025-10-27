# **Саммари**

Сделал все задания (кроме бонусного), для запуска первого и второго задания сделал отдельные requirements.txt и Dockerfile

Перед запуском залить датасет в data (download_dataset.sh)

Запуск первого

'docker build -f Dockerfile.first -t <container_name> .'

Запуск второго

'docker build -f Dockerfile.second -t <container_name> .'

Затем для обоих заданий:

'docker run --gpus all --mount type=bind,source="$(pwd)",target=/app <container_name>'

Также надо будет скачать готовые веса для FoundationStereo и поместить их в models/FoundationStereo (https://drive.google.com/drive/folders/1309JJCF8dyP9C8DJHwyEfnPYsWoGevWo)