# Информационный поиск

Материалы курса "Информационный поиск" который читается в МГТУ и других вузах осенью 2024 г.

Тут будут:

- код который показывали на семинарах
- шаблоны ДЗ
- и другие полезные материалы

## Requirements

Как самостоятельно запустить код, который показывали на семинарах?

Во-первых, нам понадобится машина с Linux, или любой другой UNIX-подобной системой, в которой:

- будет установлен python, "официальная" версия на которой мы сами проверяли код это python 3.12 (но и на более ранних скорее всего все заведется)
- будут доступны шелл (предпочтительно bash) и стандартные утилиты UNIX такие как ls и т.д.

Теперь предположим, что мы хотим запустить ноутбук _seminars/8-learning-to-rank/catboost_ltr.ipynb_ из 8-го семинара про машинное обучение ранжированию.

Сначала потребуется создать виртуальное окружение (ВНИМАНИЕ: для каждого семинара это окружение свое!).

Это делается так:

```bash
$ cd ДИРЕКТОРИЯ-В-КОТОРУЮ-ВЫ-СКЛОНИРОВАЛИ-РЕПУ-КУРСА

# Создадим папку для виртуальных окружений (если еще создавали)
$ mkdir -p .venvs

# Создадим виртуальное окружение для семинара
$ python3 -m venv .venvs/seminar-8
```

Теперь надо активировать окружение и поставить в него пакеты, перечисленные в файлике _ПАПКА-СЕМИНАРА/requirements.txt_:
```bash
# Активируем окружение
$ source .venvs/seminar-8/bin/activate

# Ставим пакеты
$ pip install -r seminars/8-learning-to-rank/requirements.txt

# Смотрим что поставилось
$ pip list
```

В результате в окружении должны стать доступны:

- все необходимые для работы питонячие библиотеки, такие как catboost или whoosh
- jupyter

Теперь просто запускаем из окружения jupyter и открываем интересный нам ноутбук в браузере:
```bash
# Запускаем jupyter
$ jupyter notebook
```
