<h3>Лаб 6: Распределение тепла с OpenACC</h3>

Создать build папку:
- cmake -Bbuild-cpu -DCMAKE_CXX_COMPILER=nvc++ -DPAR_MODE="host"
- cmake -Bbuild-multi -DCMAKE_CXX_COMPILER=nvc++ -DPAR_MODE="multicore"
- cmake -Bbuild-gpu -DCMAKE_CXX_COMPILER=nvc++ -DPAR_MODE="gpu"
</br>
(cpu - однопоточное исполнение, multi - многопоточное, gpu - на видеокарте)</br>
</br>
Далее: сборка в папке: <b>cmake --build "папка"</b></br>
И аналогичный запуск:</br>
- <b>./build-cpu/laplace</b> для CPU-onecore</br>
- <b>./build-multi/laplace</b> для CPU-multicore</br>
- <b>./build-gpu/laplace</b> для GPU</br>
