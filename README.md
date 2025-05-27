<h3>Лаб 6: Распределение тепла с OpenACC</h3>

Создать build папку:
- cmake -DACCTYPE=HOST -DCUBLAS=ON -Bbuild-cpu -S .
- cmake -DACCTYPE=MULTICORE -DCUBLAS=ON -Bbuild-multi -S .
- cmake -DACCTYPE=GPU -DCUBLAS=ON -Bbuild-gpu -S .
</br>
(cpu - однопоточное исполнение, multi - многопоточное, gpu - на видеокарте)</br>
</br>
Далее: сборка в папке: <b>cmake --build "папка"</b></br>
И аналогичный запуск:</br>
- <b>./build-cpu/laplace</b> для CPU-onecore</br>
- <b>./build-multi/laplace</b> для CPU-multicore</br>
- <b>./build-gpu/laplace</b> для GPU</br>

</br>
</br>
</br>
(Выходную матрицу в файле <b>out_matxr.txt</b> можно визуализировать с помощью matrix_parser.py)
