✅ DistilGPT2 успешно загружен!
✅ Обученная LSTM модель загружена!

================================================================================
СРАВНЕНИЕ МОДЕЛЕЙ: LSTM vs TRANSFORMER
================================================================================

1. ТЕСТИРУЕМ LSTM МОДЕЛЬ...
Validation Generation:   0%|          | 0/2501 [00:00<?, ?it/s]

==================================================
ПРИМЕР ГЕНЕРАЦИИ 1:
==================================================
ПРЕФИКС: ooh i want to learn
ОЖИДАЕМОЕ ПРОДОЛЖЕНИЕ: oh i want to learn new swear words
СГЕНЕРИРОВАННОЕ ПРОДОЛЖЕНИЕ:  the best i have to be a good

==================================================
ПРИМЕР ГЕНЕРАЦИИ 2:
==================================================
ПРЕФИКС: hoping jt wins survivor
ОЖИДАЕМОЕ ПРОДОЛЖЕНИЕ: ing jt wins survivor lt3333
СГЕНЕРИРОВАННОЕ ПРОДОЛЖЕНИЕ:  i have a good night i have to

==================================================
ПРИМЕР ГЕНЕРАЦИИ 3:
==================================================
ПРЕФИКС: yes they were indeed motherlickers the scummy kind feel dirty for even going to the interview hhh
ОЖИДАЕМОЕ ПРОДОЛЖЕНИЕ:  they were indeed motherlickers the scummy kind feel dirty for even going to the interview hhhmmmmmm frickin fake yuppies
СГЕНЕРИРОВАННОЕ ПРОДОЛЖЕНИЕ:  i have a good night i have to be a good night i have to be a good night i have to be a good night i have to
Validation Generation:   0%|          | 1/2501 [00:00<22:51,  1.82it/s]

2. ТЕСТИРУЕМ TRANSFORMER МОДЕЛЬ...

🧪 Запускаем валидацию предобученного трансформера...
Transformers Validation:   0%|          | 0/2501 [00:00<?, ?it/s]

============================================================
ПРИМЕР ГЕНЕРАЦИИ TRANSFORMER 1:
============================================================
ПРЕФИКС: yes they were indeed motherlickers the scummy kind feel dirty for even going to the interview hhh
ОЖИДАЕМОЕ ПРОДОЛЖЕНИЕ:  they were indeed motherlickers the scummy kind feel dirty for even going to the interview hhhmmmmmm frickin fake yuppies
СГЕНЕРИРОВАННОЕ ПРОДОЛЖЕНИЕ: .
I don't mean to say that I disagree with them, I just mean, they've done a lot of fucking great job getting me

============================================================
ПРИМЕР ГЕНЕРАЦИИ TRANSFORMER 2:
============================================================
ПРЕФИКС: larkvamp oh my i have no special
ОЖИДАЕМОЕ ПРОДОЛЖЕНИЕ: arkvamp oh my i have no special squid cures for stomach problems
СГЕНЕРИРОВАННОЕ ПРОДОЛЖЕНИЕ: power.

I'm a big fan of my game. I

============================================================
ПРИМЕР ГЕНЕРАЦИИ TRANSFORMER 3:
============================================================
ПРЕФИКС: i fancy this boy but he f
ОЖИДАЕМОЕ ПРОДОЛЖЕНИЕ:  fancy this boy but he fancies my twin sister
СГЕНЕРИРОВАННОЕ ПРОДОЛЖЕНИЕ: idgeted from the roof. I want to go
Transformers Validation:   0%|          | 3/2501 [00:33<7:48:14, 11.25s/it] 

✅ Проверено примеров: 100

================================================================================
РЕЗУЛЬТАТЫ СРАВНЕНИЯ:
================================================================================

📊 ROUGE МЕТРИКИ:
Метрика    LSTM     Transformer  Разница   
---------------------------------------------
rouge1     0.0876    0.1267       +0.0391
rouge2     0.0062    0.0052       -0.0010
rougeL     0.0848    0.0945       +0.0097

📏 РАЗМЕРЫ МОДЕЛЕЙ:
LSTM: 6,516,433 параметров
Transformer: 81,912,576 параметров
Отношение: 12.6x

================================================================================
РЕКОМЕНДАЦИИ ДЛЯ РАЗРАБОТЧИКОВ
================================================================================

📈 КАЧЕСТВО (ROUGE-1):
LSTM: 0.0876
...

================================================================================
ЭТАП 4 ЗАВЕРШЕН!
================================================================================
