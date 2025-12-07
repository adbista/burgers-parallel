## Jaki problem rozwiązuje równanie Burgersa?

Równanie Burgersa opisuje uproszczony przepływ płynu.

1. Pokazuje, jak prędkość „płynie” wzdłuż jednej prostej (1D).
2. Ma dwa najważniejsze elementy prawdziwej fizyki: konwekcję (przemieszczanie) i lepkość (rozmazywanie profilu).
3. Na nim sprawdza się metody numeryczne, zanim użyje się ich w trudnych symulacjach (np. pogoda, aerodynamika).

### Natura problemu

W pracy rozwiązuje się jednowymiarowe równanie Burgersa.
Zadanie nie polega tylko na tym, żeby policzyć u(x,t). Znamy rozwiązanie na końcu czasu i próbujemy odtworzyć stan początkowy, który do niego prowadzi.
Błąd liczymy jako różnicę między tym, co policzył model, a tym, co obserwujemy na końcu.

---

## Trzy metody numeryczne

### 1. Metoda Rusanova

Cechy:
- metoda niejawna: w każdym kroku czasu trzeba rozwiązać układ równań,
- korzysta z trójdiagonalnej macierzy A,
- stabilna, ale mało dokładna (pierwszy rząd).

Równoległość (trudna do realizacji):
- najtrudniejsza z trzech metod, bo trzeba rozwiązywać globalny układ równań a to słabo się skaluje,
- obliczanie b (strumieni) da się łatwo podzielić, ale solver wymusza synchronizację wszystkich procesów.

### 2. Metoda Roe

Cechy:
- metoda jawna (nie trzeba rozwiązywać układów),
- drugi rząd przestrzenny, pierwszy czasowy,
- działa na lokalnych różnicach między sąsiadami.

Równoległość (prosta do realizacji):
- każdy proces potrzebuje tylko danych od najbliższych sąsiadów (wymiana „halo”),
- żadnych globalnych operacji, więc dobrze się skaluje.

### 3. Metoda MacCormacka
Cechy:
- Predyktor – pierwszy krok. Liczy przybliżenie rozwiązania w przyszłym czasie, używając jednostronnych różnic (np. w prawo). To szybka "prognoza", która jeszcze nie jest dokładna.

- Korektor – drugi krok. Poprawia prognozę z predyktora, licząc różnice w przeciwną stronę (np. w lewo) i uśrednia oba wyniki. Dzięki temu metoda staje się dokładniejsza i bardziej symetryczna.

Równoległość (prosta do realizacji):
- potrzebna jest wymiana halo w predyktorze i korektorze,
- nie ma globalnego układu do rozwiązania, więc dobrze działa na wielu procesach.




Przebieg projektu
- definicja problemu,
- rozwiązania, krótki opis metod numerycznych
- wybrane rozwiązanie sekwencyjne: 
    - generowanie gifów dla różnych wartości parametrów
- implementacja równoległa wybranego rozwiązania - najważniejsze miejsca w kodzie i gif końcowy
    - konfiguracja bazowa
    - najważniejsze miejsca w kodzie
    - generowany gif
    - wykresy przyspieszenia, efektywności, Karpa-Flatta
- zastosowania i wnioski