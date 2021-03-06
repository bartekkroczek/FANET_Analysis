# FANET_Analysis

Efektem analizy jest olbrzymi plik ze zagegowanym szeregiem metryk dotyczących każdej osoby badanej z osobna. 
Dane zostały użytę do dalszej analizy różnic indywidualnych miedzy uczestnikami badania. 
 

## Analiza zagregowana
We wszystkich polach od TRS (włącznie) w dół uwzględniamy tylko itemy, dla których LAT &gt; 10 s (ale
itemów o LAT&lt;10 było dosłownie kilka). Część pól (np. RFRF) wylatuje (po prostu nie ma ich na liście).

* PART_ID
* SEX - plec
* AGE – wiek
* GF – czynnik gf
* WMC – czynnik wmc
* ACC – średnia poprawność w całym zadaniu
* ACC_EASY – dawne MEAN_COR_EASY itd.
* ACC_MED
* ACC_HARD
* PERC_SE_MED – proporcje błędów tylko dla warunku MEDIUM
* PERC_BE_MED
* PERC_CON_MED
* TRS – dawne RM (policzone dla wszystkich itemów)
* ERS – RM, ale policzone tylko dla błędnych itemów
* LAT_P – średni czas poprawnych reakcji w całym zadaniu
* LAT_N – średni czas błędnych reakcji w całym zadaniu  
* LAT_EASY_P – dawne MEAN_RT_EASY itd. (liczone tylko dla poprawnych)
* LAT_MED_P
* LAT_HARD_P
* LAT_EASY_N – liczone dla niepoprawnych
* LAT_MED_N
* LAT_HARD_N
* PUP_SIZE_EASY
* PUP_SIZE_MED
* PUP_SIZE_HARD
* PUP_SIZE_EASY_P
* PUP_SIZE_MED_P
* PUP_SIZE_HARD_P
* PUP_SIZE_EASY_N
* PUP_SIZE_MED_N
* PUP_SIZE_HARD_N
* NT_EASY
* NT_MED
* NT_HARD
* NT_EASY_P
* NT_MED_P
* NT_HARD_P
* NT_EASY_N
* NT_MED_N
* NT_HARD_N

* NT_PR_EASY – liczba toggli na problem (analogię, czyli to po lewej) itd.
* NT_PR_MED
* NT_PR_HARD
* NT_PR_EASY_P
* NT_PR_MED_P
* NT_PR_HARD_P
* NT_PR_EASY_N
* NT_PR_MED_N
* NT_PR_HARD_N

* NT_OP_EASY – liczba toggli na wszystkie opcje itd.
* NT_OP_MED
* NT_OP_HARD
* NT_OP_EASY_P
* NT_OP_MED_P
* NT_OP_HARD_P
* NT_OP_EASY_N
* NT_OP_MED_N
* NT_OP_HARD_N

* NT_COR_EASY – liczba toggli na poprawną opcję 
* NT_COR_MED 
* NT_COR_HARD
* NT_COR_EASY_P
* NT_COR_MED_P 
* NT_COR_HARD_P
* NT_COR_EASY_N
* NT_COR_MED_N
* NT_COR_HARD_N
* NT_ERR_EASY – liczba toggli na niepoprawne opcje
* NT_ERR_MED
* NT_ERR_HARD
* NT_ERR_EASY_P
* NT_ERR_MED_P
* NT_ERR_HARD_P
* NT_ERR_EASY_N
* NT_ERR_MED_N
* NT_ERR_HARD_N
* NT_SE_MED – liczba toggli na opcję „small error” ale tylko w warunku MEDIUM
* NT_BE_MED – liczba toggli na opcję „big error” ale tylko w warunku MEDIUM
* NT_CON_MED – liczba toggli na opcję „control” ale tylko w warunku MEDIUM

* RT_PR_EASY – dawne RTM, czyli proporcja czasu fiksacji na problemie itd.
* RT_PR_MED
* RT_PR_HARD

* RT_SE_MED – proporcja czasu fiksacji na opcjach „small error” ale tylko w warunku MEDIUM
* RT_BE_MED – proporcja czasu fiksacji na opcjach „big error” ale tylko w warunku MEDIUM
* RT_CON_MED – proporcja czasu fiksacji na opcjach „control” ale tylko w warunku MEDIUM

* RT_PR_EASY_P
* RT_PR_MED_P
* RT_PR_HARD_P
* RT_PR_EASY_N
* RT_PR_MED_N
* RT_PR_HARD_N


* RT_OP_EASY – proporcja czasu fiksacji na wszystkich opcjach
* RT_OP_MED
* RT_OP_HARD
* RT_OP_EASY_P
* RT_OP_MED_P
* RT_OP_HARD_P
* RT_OP_EASY_N
* RT_OP_MED_N
* RT_OP_HARD_N
* RT_COR_EASY – proporcja czasu fiksacji na poprawnej opcji
* RT_COR_MED
* RT_COR_HARD
* RT_COR_EASY_P
* RT_COR_MED_P
* RT_COR_HARD_P
* RT_COR_EASY_N
* RT_COR_MED_N
* RT_COR_HARD_N
* RT_ERR_EASY – proporcja czasu fiksacji na błędnych opcjach
* RT_ERR_MED
* RT_ERR_HARD
* RT_ERR_EASY_P
* RT_ERR_MED_P
* RT_ERR_HARD_P
* RT_ERR_EASY_N
* RT_ERR_MED_N
* RT_ERR_HARD_N


* DUR_PR_EASY – średni czas fiksacji na problemie (analogii, czyli tym po lewej) itd.
* DUR_PR_MED
* DUR_PR_HARD
* DUR_OP_EASY – średni czas fiksacji na wszystkich opcjach itd.
* DUR_OP_MED
* DUR_OP_HARD
* DUR_COR_EASY – stare, ale bez RV i AVG w nazwie itd.
* DUR_COR_MED 
* DUR_COR_HARD
* DUR_ERR_EASY – średni czas fiksacji uśredniony po wszystkich błędnych opcjach itd.
* DUR_ERR_MED
* DUR_ERR_HARD
* DUR_SE_MED – średni czas fiksacji na opcjach „small error” ale tylko w warunku MEDIUM
* DUR_BE_MED – średni czas fiksacji na opcjach „big error” ale tylko w warunku MEDIUM
* DUR_CON_MED – średni czas fiksacji na opcjach „control” ale tylko w warunku MEDIUM
* DUR_PR_EASY_P
* DUR_PR_MED_P
* DUR_PR_HARD_P
* DUR_OP_EASY_P
* DUR_OP_MED_P
* DUR_OP_HARD_P
* DUR_COR_EASY_P
* DUR_COR_MED_P 
* DUR_COR_HARD_P
* DUR_ERR_EASY_P
* DUR_ERR_MED_P
* DUR_ERR_HARD_P
* DUR_PR_EASY_N
* DUR_PR_MED_N
* DUR_PR_HARD_N
* DUR_OP_EASY_N
* DUR_OP_MED_N
* DUR_OP_HARD_N
* DUR_COR_EASY_N
* DUR_COR_MED_N
* DUR_COR_HARD_N
* DUR_ERR_EASY_N
* DUR_ERR_MED_N
* DUR_ERR_HARD_N
* 
* FIX_PR_EASY – sumaryczny czas fiksacji na problemie (analogii, czyli tym po lewej) itd.
* FIX_PR_MED
* FIX_PR_HARD
* FIX_OP_EASY – sumaryczny czas fiksacji na wszystkich opcjach itd.
* FIX_OP_MED
* FIX_OP_HARD
* FIX_COR_EASY – stare, ale bez RV i SUM w nazwie itd.
* FIX_COR_MED 
* FIX_COR_HARD
* FIX_ERR_EASY – sumaryczny czas fiksacji uśredniony po wszystkich błędnych opcjach itd.
* FIX_ERR_MED
* FIX_ERR_HARD
* FIX_SE_MED – sumaryczny czas fiksacji na opcjach „small error” ale tylko w warunku MEDIUM
* FIX_BE_MED – sumaryczny czas fiksacji na opcjach „big error” ale tylko w warunku MEDIUM
* FIX_CON_MED – sumaryczny czas fiksacji na opcjach „control” ale tylko w warunku MEDIUM
* FIX_PR_EASY_P
* FIX_PR_MED_P
* FIX_PR_HARD_P
* FIX_OP_EASY_P
* FIX_OP_MED_P
* FIX_OP_HARD_P
* FIX_COR_EASY_P
* FIX_COR_MED_P 
* FIX_COR_HARD_P
* FIX_ERR_EASY_P
* FIX_ERR_MED_P
* FIX_ERR_HARD_P
* FIX_PR_EASY_N
* FIX_PR_MED_N
* FIX_PR_HARD_N
* FIX_OP_EASY_N
* FIX_OP_MED_N
* FIX_OP_HARD_N
* FIX_COR_EASY_N
* FIX_COR_MED_N
* FIX_COR_HARD_N
* FIX_ERR_EASY_N
* FIX_ERR_MED_N
* FIX_ERR_HARD_N
