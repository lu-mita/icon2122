# -*- coding: utf-8 -*-

import owlready2 as owl
import os


def ontology_analyzer():
    print("ONTOLOGIA\n")
    onto = owl.get_ontology("supermarket.owl").load()

    # stampo il contenuto principale della ontologia
    print("Lista classi nella ontologia:\n")
    print(list(onto.classes()), "\n")

    print("Lista camion nella ontologia:\n")
    truck = onto.search(is_a=onto.truck)
    print(truck, "\n")

    print("Lista punti di arrivi nella ontologia:\n")
    arrivals = onto.search(is_a=onto.Point_of_arrival)
    print(arrivals, "\n")

    print("Lista punti di partenza nella ontologia:\n")
    start = onto.search(is_a=onto.Start_point)
    print(start, "\n")

    print("Lista degli alimenti nella ontologia:\n")
    food = onto.search(is_a=onto.Food_product)
    print(food, "\n")

    print("Lista dei camion hanno un costo di trasporto pari a 40€:\n")
    truck_ = onto.search(transport_cost="40€")
    print(truck_, "\n")


if __name__ == "__main__":
    ontology_analyzer()
