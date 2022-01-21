## Формат файлов с решениями

Отправлять нужно zip-файл, содержащий внутри себя единственный файл `test.jsonl`. Если вы находитесь в директории с 
файлом-ответом `test.jsonl`, то команда для создания zip-файла:
```shell
zip test.zip test.jsonl
```

Идентификаторы и сами тексты находятся в public_data/dev.jsonl

### test.jsonl

Каждая строка содержит json с ответом. Два ключа:
* `id` с идентификатором текста;
* `ners` должен соответствовать значению: Список сущностей. 
    
    Каждая сущность это список из трёх элементов:
    * индекс первого символа сущности;
    * индекс последнего символа сущности;
    * название сущности.

### Пример

```json
{"id": 4, "ners": [[0, 0, "LANGUAGE"], [0, 0, "FAMILY"], [0, 0, "LANGUAGE"]]}
{"id": 3, "ners": [[0, 0, "LANGUAGE"], [0, 0, "FAMILY"], [0, 0, "LANGUAGE"]]}
```
### Чтение и запись

Для записи таких файлов используются функции pandas 
```python
pd.DataFrame.to_json(..., lines=True, orient='records', force_ascii=False)
```
Для чтения:
```python
data = []
with open(fname, 'r') as f:
    for line in f:
        if line.strip() != '':
            data.append(json.loads(line))
```
