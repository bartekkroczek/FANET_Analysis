# Data conversions

Okulograf zapisuje dane w swoim zamkniętym formacie, czytelnym tylko przez oprogramowanie SR Research.
Chcąc analizować dane w zewnętrznych narzędziach trzeba dokonać szeregu konwersji. 
* Format zamknięty -> format ASC # Istnieje do tego (bardzo awaryjne),
 narzędzie powstałe przy wsparciu Eye-link (producent okulografu).
* Format ASC jest tekstowy, ale bardzo niewygodny do pracy. 
Społeczność stworzyła bibliotekę dla języka R (eyelinker), która ładuje pliki asc do obiektów języka R.
* asc2csv.R to skrypt korzystający z biblioteki eyelinker. W efekcie powstają wygodne do użycia pliki csv
z podziałami na zdarzenia okulograficzne. 

Cały proces był dalece mniej wygodny niż wynikało by z powyższego opisu. Właściwie każdy plik asc miał uszkodzone fragmenty,
uniemożliwiające użycie eyelinkera. Konieczne było pisanie pomocniczych skryptów przerabiających pliki asc, czy ładujących 
pliki w nieco inny sposób. W kilku skrajnych przypadkach konieczne było ręczne przerabianie plików przed/po załadowaniu.
Szczęśliwie dane udało się załadować niemal bezstratnie. 


