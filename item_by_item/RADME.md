# Pola określone dla osoby badanej (czyli dublujące się dla itemów)
* ID
* AVG_CORR
* AVG_TRS
* AVG_ERS
* GF
* WMC
# Pola określone dla poszczególnych itemów:
* KOL
* WAR
* WYB
* RESP (IF WYB = ‘COR’ THEN RESP = ‘COR’, ELSE RESP = ‘ERR’)
* LAT (zaokrąglone do pełnych sekund)
* RS (czyli ta śmieszna metryka relacyjna = liczba transformacji/liczba max. transformacji + ew. 0,02)
* PUP_SIZE
* NT_PR (podzielone przez 3, tzn. wartość per opcja)
* NT_COR
* NT_SE (podzielone przez 2, tzn. wartość per opcja)
* NT_BE (podzielone przez 2, tzn. wartość per opcja)
* NT_CON
* FIX_PR (podzielone przez 3, tzn. wartość per opcja)
* FIX_COR
* FIX_SE (podzielone przez 2, tzn. wartość per opcja)
* FIX_BE (podzielone przez 2, tzn. wartość per opcja)
* FIX_CON
* DUR_PR
* DUR_COR
* DUR_SE
* DUR_BE
* DUR_CON
Uwaga: podzielenia przez 3 są dlatego, że PRoblem składa się z 2 opcji, a przez 2, że SE i BE z 2. Każda
wartość DUR powinna być na ilorazem odpowiednich wartości FIX i NT (tzn. FIX/NT). Indeksy RT
olewamy.