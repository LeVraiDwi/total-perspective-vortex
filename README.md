# total-perspective-vortex
total-perspective-vortex 42 Project

# Dataset
## Expérimentation
Le dataset est constitué de **1 500 enregistrements** d’EEG de **1 et 2 minutes**, réalisés sur **109 volontaires**.  
Les volontaires effectuent différentes tâches motrices ou d’imagerie pendant que **64 canaux EEG** sont enregistrés à l’aide du système [BCI2000](http://www.bci2000.org).

Chaque sujet participe à **14 expérimentations** :
- **2 sessions de référence** d’une minute : yeux ouverts puis yeux fermés.
- **3 sessions de 2 minutes** pour chacune des tâches suivantes :
    1. Une cible apparaît sur la droite ou la gauche de l’écran et le sujet ouvre et ferme la main correspondante.
    2. Une cible apparaît sur la droite ou la gauche de l’écran et le sujet imagine ouvrir et fermer la main correspondante.
    3. Une cible apparaît en haut ou en bas de l’écran et le sujet ouvre et ferme soit ses deux mains (haut de l’écran), soit ses deux pieds (bas de l’écran).
    4. Une cible apparaît en haut ou en bas de l’écran et le sujet imagine ouvrir et fermer soit ses deux mains (haut de l’écran), soit ses deux pieds (bas de l’écran).

### Organisation de la session :
- session de référence yeux ouverts
- session de référence yeux fermés
- tâche 1
- tâche 2
- tâche 3
- tâche 4
- tâche 1
- tâche 2
- tâche 3
- tâche 4
- tâche 1
- tâche 2
- tâche 3
- tâche 4

Les données sont stockées dans des fichiers au format [EDF+](https://www.edfplus.info/specs/edfplus.html) contenant les **64 canaux**, chacun échantillonné à **160 échantillons par seconde**, ainsi qu’un **canal d’annotation**.  
Ce canal d’annotation permet de connaître :
- le type de tâche (main gauche, main droite, imagerie, repos, etc.)
- les périodes de sessions de référence

Ce canal d’annotation est extrait dans un fichier d’annotation (`.event`) utilisé par le framework **PhysioToolkit**.  
Il y a **trois types d’annotations** :
- **T0** : repos
- **T1** : main gauche (tâches 1/2) ou les deux mains (tâches 3/4)
- **T2** : main droite (tâches 1/2) ou les deux pieds (tâches 3/4)

## Montage
Les signaux EEG ont été enregistrés à partir de **64 électrodes** selon le système international **10-10** (à l’exclusion des électrodes Nz, F9, F10, FT9, FT10, A1, A2, TP9, TP10, P9 et P10)  
([schéma](https://physionet.org/content/eegmmidb/1.0.0/64_channel_sharbrough.pdf)).

Le numéro indiqué sous chaque nom d’électrode (1–64) correspond à l’ordre dans lequel elle apparaît dans l’enregistrement (0–63).

## EDF+ (European Data Format)
extension du format EDF, Standart qui permet d’enregistrer la plupart des données EEG, PSG, ECG, EMG ainsi que des potentiels évoqués, qui ne peuvent pas être stockées dans les systèmes d’information hospitaliers courants.
Un fichier EDF+ contient :
- des signaux multicanaux (par exemple 64 canaux EEG),
- une fréquence d’échantillonnage définie pour chaque canal,
- un canal d’annotation permettant de stocker des événements synchronisés dans le temps (début/fin de tâches, stimuli, repos, etc.),
- des métadonnées décrivant l’enregistrement (sujet, durée, capteurs, unités).

# Glossaire
**EEG** : Electroencephalogram

# Références et citations
## PhysioNet
[Link to data](https://physionet.org/content/eegmmidb/1.0.0/)

### Original publication
[Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. *BCI2000: A General-Purpose Brain-Computer Interface (BCI) System*. IEEE Transactions on Biomedical Engineering 51(6):1034–1043, 2004.](https://pubmed.ncbi.nlm.nih.gov/15188875/)

---

**APA**  
Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). *PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals*. Circulation [Online], 101(23), e215–e220. RRID:SCR_007345.

**MLA**  
Goldberger, A., et al. “PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.” *Circulation* [Online], vol. 101, no. 23, 2000, pp. e215–e220. RRID:SCR_007345.

**CHICAGO**  
Goldberger, A., L. Amaral, L. Glass, J. Hausdorff, P. C. Ivanov, R. Mark, J. E. Mietus, G. B. Moody, C. K. Peng, and H. E. Stanley. “PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.” *Circulation* [Online] 101, no. 23 (2000): e215–e220. RRID:SCR_007345.

**HARVARD**  
Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P.C., Mark, R., Mietus, J.E., Moody, G.B., Peng, C.K. and Stanley, H.E., 2000. PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation* [Online], 101(23), pp. e215–e220. RRID:SCR_007345.

**VANCOUVER**  
Goldberger A, Amaral L, Glass L, Hausdorff J, Ivanov PC, Mark R, Mietus JE, Moody GB, Peng CK, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation* [Online]. 2000;101(23):e215–e220. RRID:SCR_007345.
