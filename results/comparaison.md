# Comparaison des modèles quantifiés MNIST

Évaluation de 10 000 images de test. Chaque checkpoint correspond à 5 époques.
Résultats de l'exécution précédente ; le checkpoint INT32 n'est plus présent.

| Modèle | Précision | Poids + biais théoriques | Paramètres PyTorch | Fichier .pth |
|---|---:|---:|---:|---:|
| INT32 | 98.57 % | 35.54 Kio | 35.55 Kio | 57.82 Kio |
| INT8 | 98.27 % | 8.98 Kio | 35.55 Kio | 57.80 Kio |
| INT4 | 98.24 % | 4.56 Kio | 35.55 Kio | 57.80 Kio |
| INT2 | 97.12 % | 2.35 Kio | 35.55 Kio | 57.80 Kio |

1 Kio = 1024 octets. Taille théorique : poids compactés selon le nombre de bits et biais FP32, hors échelles, activations et mémoire de travail.
Brevitas conserve des paramètres flottants : le fichier .pth ne contient pas un export entier compacté pour FPGA.
La mémoire PyTorch indiquée ne compte que les paramètres, sans les buffers, gradients, états Adam ou activations.
Ces modèles ont été entraînés séparément ; une seule mesure par modèle ne permet pas de séparer l’effet des bits de la variabilité de l’entraînement.

Pour relancer une mesure depuis la racine :

```bash
python src/evaluate.py --bits 8
```
