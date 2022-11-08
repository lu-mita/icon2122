# Progetto ICON 2021/2022

## Vehicle Routing
Un problema che può essere formulato come CSP è il Vehicle Routing Problem.
Minimizzare la tratta percorsa dai veicoli (soft constraint), rispettando le finestre temporali e i limiti di capacità dei veicoli (hard constraints) e raggiungendo tutti i punti di consegna.

La libreria utilizzata per risolvere questo problem è or-tools. L'algoritmo risolve il CSP creado prima delle assegnazioni parziali
e ordinando i valori delle variabili cercando l'arco meno costoso (in questo caso la distanza percorsa)
controllando di volta in volta il rispetto dei vincoli forti. Una volta raggiunta una assegnazione completa l'algoritmo prosegue cercando di ottimizzare l'assegnazione tramite ricerca locale. E' possibile indicare un limite temporale entro il quale far eseguire la ricerca, anche accontentandosi anche di una soluzione subottimale.
Datogli abbastanza tempo l'algoritmo convergerà ad una soluzione ottimale.


## Recommendation Systems

