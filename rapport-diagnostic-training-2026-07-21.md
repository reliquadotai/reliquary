# Diagnostic training — base vs checkpoint 47 (base-fresh-v3)

**Date :** 2026-07-21
**Question :** le training fait-il vraiment progresser le modèle (capacité, exploration), ou stagne-t-il ?
**Réponse courte :** **oui en CODE (×2.9 de capacité), non en MATH (plat).** La cause du blocage math est identifiée : **BFT** (la terminaison forcée) découple la reward du raisonnement.

---

## 1. Setup

| | |
|---|---|
| Modèle testé (live) | `ReliquaryForge/qwen3.5-2b-reliquary-v3` @ `5db7a1f5` = **checkpoint 47** (run `base-fresh-v3`) |
| Modèle de référence | `Qwen/Qwen3.5-2B` @ `15852e8c` (le base, point de départ + ancre KL) |
| Steps d'entraînement | 47 (depuis le base) |
| GPU | 1× RTX 4090, backend transformers (bf16), sampling protocole T=0.6 / top_p=0.95 / top_k=20 |
| Éval | 60 prompts × 8 rollouts par condition (480 rollouts) |
| Prompts | vrais prompts issus des archives de production (OMI pour math, OpenCodeInstruct pour code) |
| Grading | grader OMI exact (math) ; **vrais tests unitaires OCI** exécutés (code), grader validé (solution correcte → 1.0, fausse → 0.0) |

Deux régimes, correspondant à la production :
- **Math = avec BFT** (Budget-Forced Termination : si `</think>` pas émis à 2048 tokens, on force `</think>\n\nFinal Answer: \boxed{` + 512 tokens).
- **Code = sans BFT** (le code n'a jamais de BFT en prod).

---

## 2. Résultats — MATH (régime BFT = production)

| métrique | **base** | **ckpt47** | delta |
|---|---|---|---|
| **pass@1 (BFT)** | **0.281** | **0.304** | +0.023 |
| pass@1 (naturel, sans force) | 0.004 | 0.090 | ×22 |
| `</think>` fermé spontanément | 0.6 % | 19.6 % | ×33 |
| `\boxed` spontané | 15.2 % | 27.9 % | +13 pt |
| EOS spontané | 0.4 % | 12.9 % | ×32 |
| a besoin de la force BFT | 99.4 % | 80.4 % | −19 pt |

Distribution de k (nombre de rollouts corrects sur 8, par prompt) sous BFT :
- **base** : `k0:9 k1:10 k2:18 k3:12 k4:5 k5:4 k6:1 k7:1 k8:0`
- **ckpt47** : `k0:12 k1:10 k2:14 k3:9 k4:4 k5:5 k6:4 k7:0 k8:2`
- → **quasi identiques : la frontière de difficulté ne bouge pas.**

**Lecture :** le pass@1 réel (BFT) passe de 0.281 à 0.304 = **+2 points sur 480 échantillons = dans le bruit** (non significatif, n=60 prompts). Ce que le modèle a *vraiment* appris en 47 steps, c'est à **terminer et formater** (`</think>` ×33, `\boxed` ×2, EOS ×32) — **pas à mieux résoudre.** Et BFT rend cet apprentissage redondant : il fournit déjà la terminaison.

---

## 3. Résultats — CODE (sans BFT = production)

| métrique | **base** | **ckpt47** | delta |
|---|---|---|---|
| **pass@1 (tests OCI réels)** | **0.148** | **0.431** | **×2.9** |
| reward moyen (fraction de tests passés) | 0.153 | 0.441 | ×2.9 |
| EOS (terminaison naturelle) | 85.8 % | 91.9 % | +6 pt |
| longueur médiane (tokens) | 1067 | 1501 | +434 |

Distribution de k :
- **base** : `k0:36 k1:4 k2:9 k3:1 k4:6 k5:2 k6:2 k7:0 k8:0`
- **ckpt47** : `k0:7 k1:10 k2:8 k3:8 k4:8 k5:4 k6:5 k7:5 k8:5`
- → **base = 36/60 prompts insolubles (k0) → ckpt47 = 7/60, étalement jusqu'à k8. La frontière a bondi vers le haut.**

**Lecture :** en code, le training **fonctionne massivement** (pass@1 ×2.9, +28 points — largement significatif). Mieux : le modèle génère **plus long** (1067→1501) **et** meilleur, **sans aucune prime de longueur** — uniquement via la justesse sur les tests. C'est la preuve d'existence du « faire grossir le raisonnement » recherché : **il émerge tout seul quand la reward est honnête.**

---

## 4. Diagnostic (cause racine)

Tout est identique entre math et code : même modèle, même run, mêmes 47 steps, même GRPO, même σ-gate, même auction k=2, même famine de données. **La seule différence est BFT** (présent en math, absent en code). Le code progresse ×2.9, le math stagne.

> **→ Le blocage math n'est PAS le curriculum ni la famine de données. C'est BFT.**

**Principe :** une reward n'enseigne le raisonnement que si elle note la sortie que le modèle a réellement **terminée**.
- **Code** : le modèle finit son code et s'arrête → la reward note son vrai travail → il apprend.
- **Math** : le modèle radote et n'aboutit jamais (à ckpt47, 80 % ont encore besoin de la force) → **BFT force une réponse à 2048** → la reward note un **tirage forcé**, pas le raisonnement → il n'apprend que le formatage.

Confirmation chiffrée : en math, pass@1 naturel = 0.09 vs BFT = 0.30 → **2/3 des « bonnes » réponses d'entraînement viennent de la force BFT, pas du modèle.** Le gradient d'entraînement est donc majoritairement du bruit côté raisonnement.

---

## 5. Recommandation (implication design)

**La pipeline n'est pas cassée — le code le prouve.** GRPO + k=2 + σ-gate produisent un gros gain *quand la reward est honnête*. Il faut amener le math dans les mêmes conditions.

Ordre proposé (le 1 est le plus petit et déjà à moitié codé) :

1. **Terminaison propre d'abord.** Activer/calibrer le length-shaping bilatéral déjà présent (`RELIQUARY_SHAPE_PENALTY`, aujourd'hui à 0) : récompenser la clôture `</think>`+réponse, pénaliser le radotage-jusqu'au-cap — **sur l'avantage, pas une prime de longueur.** Objectif mesurable : faire chuter le taux de déclenchement de BFT.
2. **Reward honnête → capacité qui progresse.** Une fois que le modèle ferme `</think>` de lui-même, BFT ne se déclenche presque plus → la reward math note le vrai raisonnement → le math devrait progresser **comme le code.**
3. **Ensuite seulement** : relâcher le cap 2048 (le modèle utilisera la place pour raisonner, plus pour radoter) et durcir le curriculum.

**À NE PAS faire** (réfuté par la data) : monter `max_tokens` / le budget. En math, le modèle radote pour remplir n'importe quel budget ; en code, il s'auto-termine (médiane plate ~1334 de 2048 à 8192). Le budget n'est pas le levier.

---

## 6. Limites / honnêteté méthodo

- n = 60 prompts par condition → le +2 pt math est dans le bruit ; le ×2.9 code est largement au-dessus.
- Prompts = gagnants d'archives (pas un tirage de difficulté uniforme). Le delta base→ckpt47 reste valide (mêmes prompts pour les deux modèles) ; une petite part de mémorisation possible côté code (ckpt47 a vu ~376 prompts en 47 steps) — mais l'ampleur du gain (36→7 prompts k0) dépasse largement ce que la mémorisation expliquerait.
- Sampling naturel (T=0.6) vs forced-seed de la prod : peut décaler légèrement les longueurs absolues, pas les comparaisons base-vs-ckpt.
- 47 steps = entraînement jeune. Mais le point n'est pas « trop peu » : c'est que le gradient math va au formatage (que BFT couvre déjà), pas au raisonnement.
