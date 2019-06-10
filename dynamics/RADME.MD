# Analiza dynamiki fiksacji

Główny wątek analiz. 

Analiza dotyczy zawsze N pojednyczych triali, przy czym N musi być odpowiednio duże, aby analiza miała sens. Każdy trial i ma latencję odpowiedzi Li w s (u nas od 0 do 120 s) oraz odpowiedź Oi (o wartościach - opcjach odpowiedzi od D1 do D6). W każdym trialu fiksacje są możliwe albo na PRoblemie albo na sześciu OPcjach. Każda opcja ma swoją wartość metryki relacyjnej RM, o której pisałem wcześniej (RM = stosunek liczby zmienionych poprawnie cech w danej opcji do liczby takich cech w opcji D1, dla dodatkowych cech w D2 dodajemy .02). W tej analizie dynamicznej, żeby wszystko uprościć, będziemy się posługiwać tylko RM zamiast COR, SE, BE, CON.

1. Ustalamy jakie jest maksymalne Lmax dla którego wciąż jest nie mniej niż Z triali (przyjmijmy Z = 10, tzn. żeby uniknąć przypadkowych wyników dla kilku najdłuższych prób).
2. Dla każdej kolejnej sekundy Lx, począwszy od ustawionej arbitralnie Lmin (na razie przyjmijmy 0 s, ale myślę że to będzie 10 s) do Lmax, ustalamy ile triali Kx spośród N się jeszcze nie skończyło w tej x-ej sekundzie (tzn. dla ilu triali Li => Lx).
3. Dla każdej sekundy Lx, dla każdego trialu i spośród Kx, ustalamy najdłuższą fiksację przypadającą na tę sekundę (w większości przypadków będzie jedna fiksacja).
4. Jeśli jest to fiksacja na PR, to zmienna "fiksacja na opcjach" FOix = 0, jeśli fiksacja jest na OP to FOix = 1. Jeśli FOix = 1, to RMix = RM opcji na której jest fiksacja (od 0 dla D6 do 1 dla D1). Jeśli FOix = 0, to RMix = -1.
5. Dla każdej sekundy Lx sumujemy po i wartości FOix, dostając FOx, które dzielimy przez Kx, dostając PROP_FOx, czyli proporcję fiksacji na opcjach wśród triali Kx w sekundzie x.
6. Dla każdej sekundy Lx sumujemy nieujemne wartości RMix, dostając RMx, które dzielimy przez liczbę prób dla których RMix jest nieujemne, dostając AVG_RMx, czyli średnie RM opcji na których były fiksacje (jeśli były na opcjach a nie na problemie) w sekundzie x. Oczywiście, jeśli w danej x w żadnym i nie patrzono się na OP, to AVG_RMx pozostaje nieustalone. 
7. Na wykresie o osi poziomej przedstawiającej sekundy od Lmin do Lmax, a osi pionowej o skali od 0 do max(max(PROP_FOx), max(AVG_RMx)), czyli zależnie które z nich jest większe, rysujemy linię łączącą kolejne wartości PROP_FOx (np. jako przerywaną) oraz linię łączącą kolejne wartości AVG_RMx (jako ciągłą).
I tyle. Czyli wykres ten pokazuje jak zmieniała się alokacja fiksacji pomiędzy problem a opcje oraz jak zmieniała się relacyjność opcji na których się fiksowano. 

Teraz jak ta ogólna analiza miała by wyglądać w naszym artykule. W każdej z analiz bierzemy wszystkie próby, zarówno poprawne i niepoprawne, wrzucając do jednego worka.
## Analiza A - dynamika w funkcji wybranej odpowiedzi
1. Dzielimy triale na takie w których wybrano odpowiedź albo poprawną albo niepoprawną
2. Dla każdego z dwóch zbiorów osobno zapuszczamy analizę dynamiki fiksacji
3. Rysujemy każdy z dwóch wykresów pod sobą, oraz wykres dla połączonych triali.
## Analiza B - czy czas rozwiązywania triala wpływa jakoś na dynamikę?
1. Dzielimy triale na trwające do 40 s, 41-80 s, 81-120 s.
2. Dla każdego z trzech zbiorów osobno zapuszczamy analizę dynamiki fiksacji
3. Rysujemy każdy z (coraz dłuższych) wykresów kolejno pod sobą.
## Analiza C - zakładam, że czas triala nie wpływa aż tak strasznie na dynamikę, więc już go nie będziemy uwzględniać, a sprawdzimy inne zmienne.
1. Dzielimy triale na EASY, MEDIUM, HARD
2. Dla każdego z trzech zbiorów osobno zapuszczamy analizę dynamiki fiksacji
3. Rysujemy każdy z (mniej więcej równej długości) wykresów kolejno pod sobą.
## Analiza D - dynamika w funkcji WMC
1. Dzielimy ludzi na tercyle w funkcji wartości ich czynnika WMC (low, medium, high)
2. Dla każdej grupy ludzi osobno zapuszczamy analizę dynamiki fiksacji
3. Rysujemy każdy z trzech (prawdopodobnie o lekko rosnącej długości) wykresów pod sobą, 

# II.Analiza dynamiki sakad - tu znów wracamy do COR, SE, BE, CON

1. Dla określonego zbioru triali zliczamy w okresie od Lmin do Lmax przejścia (sakady, nieważne w którą stronę) pomiędzy każdą kombinacją pięciu kategorii: A, B, C, COR, ERR (wszystkie opcje niepoprawne) i dzielimy przez łączną liczbę znalezionych przejść. 
2. Rysujemy trójkąt równoboczny o boku = 4, kolorami albo wielkością koła (jeszcze zobaczymy co lepiej widać) pokazując w każdej jego kratce proporcję danego typu przejść wśród wszystkich przejść. Trójkąt w takiej orientacji:

A-B     

A-C       B-C     
 
A-COR  B-COR C-COR

A-ERR  B-ERR  C-ERR  COR-ERR

## Analiza E
przejścia w funkcji albo poprawnej albo niepoprawnej odpowiedzi - rysujemy 3 trójkąty osobno dla każdej kategorii odpowiedzi oraz łącznie.

## Analiza F
znów dzielimy triale na do 40 s, 41-80 s, 81-120 s i rysujemy trójkąt osobno dla każdego z trzech zbiorów triali (w linii po sobie).

## Analiza G
przejścia w funkcji warunku - rysujemy 3 trójkąty dla EASY, MEDIUM, HARD.

## Analiza H
przejścia w funkcji WMC - rysujemy trzy trójkąty dla tercyli jak w analizie D.
