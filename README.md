# HammAI & NavAI

### HTML and clear text

**для запуска**:
1) создайте файл  `.env` в корне репозитория и запишите в него следующие переменные:
```
BACKEND_PORT=8005
FRONTEND_PORT=8006
```
2) запустите сборку контейнера: (в первый запуск загрузка модели займет некоторое время)
```
docker-compose --env-file .env up --build
```
>Если будет ошибка при скачивании - просто перезапустите команду, иногда google хулиганит
>Протестировано на Mac OS и Linux
3) Постучаться в бекенд: 
```commandline
http://localhost:BACKEND_PORT
```
4) Постучаться в фронтенд: 
```commandline
http://localhost:FRONTEND_PORT
```
>Если тестируете не локально, то ip можно найти в команде `ifconfig`
5) [видео с работой сервиса](https://drive.google.com/file/d/1nKVEntGDkCQlUuKkcJtc8PN9KXBQnc4y/view?usp=sharing) and [презентация](https://docs.google.com/presentation/d/1SYX7P5-FobwfJRxwEDB0raLcvawMMB6rwvf_wDb1nKQ/edit#slide=id.g280424bac3a_0_81)
   
