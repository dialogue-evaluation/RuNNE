# Соревнование RuNNE: извлечение именованных сущностей в few-shot режиме
## Введение
Извлечение именованных сущностей – одна из самых востребованных на практике задач извлечения информации – предполагает поиск в тексте упоминаний имен, организаций, топонимов и других сущностей.  Соревнование RuNNE посвящено задаче извлечения вложенных именованных сущностей. Разметка данных допускает следующие случаи: внутри одной именованной сущности находится другая именованная сущность. Так, например в сущность класса Organization “Московский драматический театр имени М. Н. Ермоловой” вложена сущность типа Person – “М. Н. Ермоловой”. 


## Данные
Соревнование проводится на материале корпуса NEREL [1], собранного из новостных текстов WikiNews на русском языке. В корпусе NEREL представлено 29 классов различных сущностей, а глубина вложенности сущностей достигает 6 уровней разметки. 

Данные предоставляются участникам в виде размеченных документов. Формат разметки – BRAT.


## Постановка задачи 
В рамках соревнования RuNNE мы предлагаем  участникам рассмотреть few shot  постановку задачи. 
Задача предполагает извлечение вложенных именованных сущностей,
В обучающем множестве большая часть типов именованных сущностей  встречается достаточно часто, а некоторое количество специально отобранных типов – встречается всего несколько раз, 
В тестовом множестве все типы сущностей представлены одинаково.

Таким образом, участникам предстоит разработать модели извлечения вложенных именованных сущностей, поддерживающие few-shot режим. 

## Оценка соревнования
В качестве метрики качества в соревновании RuNNE используется макро усреднение F1-меры в двух вариантах:  по классам известных сущностей (общая постановка задачи извлечения вложенных именованных сущностей) и по классам новых именованных сущностей (few-shot постановка). 



## Правила участия 
* Участникам соревнования разрешается использовать любые дополнительные материалы и любые предобученные модели, за исключением непосредственной разметки тестового множества. 
* Участники могут самостоятельно разметить дополнительные данные в соответствии с опубликованными инструкциями. При этом, организаторы соревнования будут просить участников опубликовать новые размеченные данные в открытом доступе. 

## Полезные ссылки
* платформа для отладки: https://codalab.lisn.upsaclay.fr/competitions/1142 
* платформа для тестирования: https://codalab.lisn.upsaclay.fr/competitions/1863 
* Github: https://github.com/dialogue-evaluation/RuNNE
* Tlg: t.me/deval_RuNNE

## Ключевые даты 
* 29 декабря 2021 – публикация обучающих данных 
* 7 февраля 2022 –  публикация тестовых данных
* 24 февраля 2022 по AoE – закрытие тестирования 
* 25 марта – завершаем прием статей 


## Организаторы
* Наталья Лукашевич (МГУ)
* Екатерина Артемова (Huawei, НИУ ВШЭ)
* Татьяна Батура (НГУ, ИСИ СО РАН)
* Павел Браславский (НИУ ВШЭ, УРФУ)
* Владимир Иванов (Иннополис)
* Елена Тутубалина (Sber AI, НИУ ВШЭ)


## Публикации

[1] Loukachevitch N. et al. [NEREL: a Russian information extraction dataset with rich annotation for nested entities, relations, and wikidata entity links](https://link.springer.com/article/10.1007/s10579-023-09674-z). Language Resources and Evaluation (2023). https://doi.org/10.1007/s10579-023-09674-z

```
    @article{loukachevitch2023nerel,
      title={NEREL: a Russian information extraction dataset with rich annotation for nested entities, relations, and wikidata entity links},
      author={Loukachevitch, Natalia and Artemova, Ekaterina and Batura, Tatiana and Braslavski, Pavel and Ivanov, Vladimir and Manandhar, Suresh and Pugachev, Alexander and Rozhkov, Igor and Shelmanov, Artem and Tutubalina, Elena and others},
      journal={Language Resources and Evaluation},
      pages={1--37},
      year={2023},
      publisher={Springer}
    }
```

[2] Loukachevitch N., Artemova E., Batura T., Braslavski P., Denisov I., Ivanov V., Manandhar S., Pugachev A., Tutubalina E. [NEREL: A Russian Dataset with Nested Named Entities, Relations and Events](https://acl-bg.org/proceedings/2021/RANLP%202021/pdf/2021.ranlp-1.100.pdf). Proceedings of RANLP. 2021. pp. 880–889.

```

    @inproceedings{loukachevitch2021nerel,
      title={{NEREL: A Russian} Dataset with Nested Named Entities, Relations and Events},
      author={Loukachevitch, Natalia and Artemova, Ekaterina and Batura, Tatiana and Braslavski, Pavel and Denisov, Ilia and Ivanov, Vladimir and Manandhar, Suresh and Pugachev, Alexander and Tutubalina, Elena},
      booktitle={Proceedings of RANLP},
      pages={876--885},
      year={2021}
    }

```

[3] Artemova, E., Zmeev, M., Loukachevitch, N., Rozhkov, I., Batura, T., Ivanov, V., & Tutubalina, E. RuNNE-2022 Shared Task: Recognizing Nested Named Entities. Komp'juternaja Lingvistika i Intellektual'nye Tehnologii. 2022. pp. 33-41.



```
@article{artemova2022runne,
 author = {Artemova, Ekaterina and Zmeev, Maxim and Loukachevitch, Natalia and Rozhkov, Igor and Batura, Tatiana and Ivanov, Vladimir and Tutubalina, Elena},
 title = {RuNNE-2022 Shared Task: Recognizing Nested Named Entities},
 year = {2022},
 journal = {Komp'juternaja Lingvistika i Intellektual'nye Tehnologii},
 volume = {2022},
 number = {21},
 pages = {33 – 41},
 doi = {10.28995/2075-7182-2022-21-33-41}
}
```
