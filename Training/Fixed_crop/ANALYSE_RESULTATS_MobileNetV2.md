# Analyse des Résultats - MobileNetV2 vs CNN Simple

## 📊 Résultats de la Validation Croisée (5-Fold)

### MobileNetV2
- **Val Loss moyenne** : 0.4157 ± 0.1721
- **Val Accuracy moyenne** : 83.95% ± 8.15%
- **Détails par fold** :
  - Fold 1: Loss=0.5607, Acc=76.65%, Epoch=5
  - Fold 2: Loss=0.1843, Acc=94.61%, Epoch=18 ⭐ (meilleur)
  - Fold 3: Loss=0.3414, Acc=88.02%, Epoch=15
  - Fold 4: Loss=0.3303, Acc=88.02%, Epoch=8
  - Fold 5: Loss=0.6620, Acc=72.46%, Epoch=16

### CNN Simple (référence)
- **Val Loss moyenne** : 0.2487 ± 0.0684
- **Val Accuracy moyenne** : ~91% (estimé)

## 🎯 Résultats sur le Test Set Final

### MobileNetV2
- **Accuracy globale** : **93.55%**
- **AUC** : **0.9963** (excellent)
- **F1-Score macro** : 0.9296
- **F1-Score weighted** : 0.9368

### Métriques par classe (MobileNetV2)

| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|----------|
| **Abnormal_Heartbeat** | 1.0000 | 0.8696 | 0.9302 | 23 |
| **History_MI** | 0.7727 | 1.0000 | 0.8718 | 17 |
| **Myocardial_Infarction** | 1.0000 | 0.8750 | 0.9333 | 24 |
| **Normal** | 0.9667 | 1.0000 | 0.9831 | 29 |

### Matrice de confusion (MobileNetV2)
```
                    Prédiction
                  AHB  HMI  MI  N
Vraie  AHB        20   2    0   1
       HMI         0  17    0   0
       MI          0   3   21   0
       N           0   0    0  29
```

**Erreurs identifiées** : 6 erreurs au total
- Myocardial_Infarction → History_MI : 3 erreurs
- Abnormal_Heartbeat → History_MI : 2 erreurs
- Abnormal_Heartbeat → Normal : 1 erreur

## 🔍 Observations et Constatations

### ✅ Points Positifs

1. **Performance globale excellente** :
   - Accuracy de 93.55% sur le test set
   - AUC de 0.9963 (quasi-parfait)
   - F1-Score macro de 0.9296

2. **Gestion du déséquilibre efficace** :
   - Les poids de classe ont été calculés correctement
   - History_MI (classe la plus sous-représentée avec 18.5%) a le poids le plus élevé (1.3488)
   - Normal (classe la plus représentée avec 30.6%) a le poids le plus faible (0.8169)

3. **Classe "Normal" excellente** :
   - Precision : 96.67%
   - Recall : 100%
   - F1-Score : 98.31%
   - Aucune confusion avec les autres classes

4. **Early Stopping fonctionne** :
   - Les modèles s'arrêtent automatiquement entre 5 et 18 epochs
   - Évite le surapprentissage

### ⚠️ Points d'Attention

1. **Variabilité importante en cross-validation** :
   - Écart-type élevé : ±8.15% pour l'accuracy
   - Fold 1 : 76.65% (faible)
   - Fold 5 : 72.46% (très faible)
   - Fold 2 : 94.61% (excellent)
   - **Cela suggère une sensibilité aux splits de données**

2. **Classe "History_MI" problématique** :
   - Precision : 77.27% (la plus faible)
   - Recall : 100% (trop de faux positifs)
   - **Le modèle confond souvent d'autres classes avec History_MI** :
     - 3 cas de Myocardial_Infarction classés comme History_MI
     - 2 cas d'Abnormal_Heartbeat classés comme History_MI
   - **Cela peut être dû à la similarité médicale entre ces conditions**

3. **Classe "Abnormal_Heartbeat"** :
   - Precision : 100% (parfaite)
   - Recall : 86.96% (manque quelques cas)
   - 3 erreurs : 2 classées comme History_MI, 1 comme Normal

4. **Classe "Myocardial_Infarction"** :
   - Precision : 100% (parfaite)
   - Recall : 87.50% (manque quelques cas)
   - 3 erreurs : toutes classées comme History_MI

### 🔬 Analyse des Erreurs

**Pattern d'erreurs observé** :
- **History_MI semble être une "classe piège"** qui attire les erreurs
- Les confusions principales :
  1. Myocardial_Infarction → History_MI (3 cas)
  2. Abnormal_Heartbeat → History_MI (2 cas)
  3. Abnormal_Heartbeat → Normal (1 cas)

**Raisons possibles** :
1. **Similarité médicale** : History_MI et Myocardial_Infarction sont des conditions cardiaques liées
2. **Déséquilibre** : History_MI est la classe la plus sous-représentée (172 images vs 233-284 pour les autres)
3. **Caractéristiques visuelles** : Les patterns ECG peuvent être similaires entre ces conditions

### 📈 Comparaison avec CNN Simple

| Métrique | MobileNetV2 | CNN Simple |
|----------|-------------|------------|
| **Test Accuracy** | 93.55% | 97.85% |
| **CV Val Loss** | 0.4157 ± 0.1721 | 0.2487 ± 0.0684 |
| **CV Val Acc** | 83.95% ± 8.15% | ~91% |
| **Paramètres** | 2,228,996 | ~500K (estimé) |
| **Complexité** | Élevée | Faible |

**Constats** :
- Le CNN simple semble **légèrement meilleur** sur ce dataset spécifique
- MobileNetV2 a une **variabilité plus élevée** (écart-type plus grand)
- MobileNetV2 est **plus complexe** mais n'apporte pas d'avantage clair ici

### 💡 Recommandations

1. **Pour améliorer History_MI** :
   - Augmenter le nombre d'échantillons pour cette classe
   - Utiliser des techniques de data augmentation plus agressives
   - Augmenter le poids de classe pour History_MI encore plus
   - Considérer un modèle d'ensemble

2. **Pour réduire la variabilité** :
   - Augmenter le nombre de folds (10-fold au lieu de 5)
   - Utiliser plus de régularisation (dropout plus élevé)
   - Réduire le learning rate initial

3. **Pour améliorer globalement** :
   - Considérer un fine-tuning plus poussé des couches pré-entraînées
   - Essayer d'autres architectures (ResNet, EfficientNet)
   - Utiliser des techniques de focal loss pour mieux gérer les classes difficiles

4. **Analyse médicale** :
   - Les confusions entre History_MI et Myocardial_Infarction sont médicalement compréhensibles
   - Il pourrait être pertinent de combiner ces classes ou d'utiliser une hiérarchie de classification

## 📝 Conclusion

Le modèle MobileNetV2 atteint une **performance très bonne (93.55%)** mais présente :
- ✅ **Forces** : Excellent AUC, bonne gestion du déséquilibre, classe Normal parfaite
- ⚠️ **Faiblesses** : Variabilité élevée en CV, confusions avec History_MI, performance légèrement inférieure au CNN simple

**Le modèle est utilisable en production** mais nécessiterait des améliorations pour les cas cliniques critiques, notamment pour mieux distinguer History_MI des autres conditions cardiaques.

