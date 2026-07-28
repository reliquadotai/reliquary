# Pourquoi notre modèle apprend à *terminer* mais pas à *raisonner mieux* (GRPO, et le contraste avec DeepSeek-R1)

**Date : 2026-07-22** — évals GPU sur held-out (envs subnet OMI/OCI), base Qwen3.5-2B vs checkpoint live.

## TL;DR

Le training (GRPO / RL outcome-reward) fait **une** chose de façon spectaculaire : il apprend au modèle à **conclure son raisonnement** (fermer `</think>`, poser sa réponse / émettre son code au lieu de radoter). Il **n'améliore quasiment pas la capacité** (résolution math, qualité de code) et **n'allonge pas le raisonnement**. R1 (DeepSeek) a fait l'inverse — capacité et longueur de CoT qui explosent. La raison tient en une phrase :

> **R1 a allongé son raisonnement parce que, sur un gros base, "raisonner plus long ⇒ être plus souvent correct" — donc le gradient de correction sélectionnait les longues chaînes. Sur notre 2B, cette corrélation n'existe pas (long = radotage), donc il n'y a rien à allonger ni à approfondir.**

---

## 1. Ce qu'on mesure

Sur les **envs du subnet** (held-out, 0 overlap), base → checkpoint :

**Code (OCI)** : pass@1 0.353 → 0.539. MAIS en décomposant les échecs du base : le gain vient **presque entièrement** de la chute des générations *sans code émis* (« nocode » : radote dans le thinking, coupé au budget) — 35% → 5%. La qualité du code écrit, elle, bouge à peine.

**Math (OMI, régime BFT)** :
- Résolution brute (pass@1 BFT, qui force une réponse quoi qu'il arrive) : 0.703 → 0.647 → **plat, voire léger recul**.
- Terminaison : `</think>` fermé seul 11% → 63% ; pass@1 *naturel* (sans force) 0.18 → 0.57.

**Longueur (code)** : médiane 1514 → 1588 (**plate**) ; moyenne 2309 → 2088 (**en baisse**, parce que la moyenne du base était gonflée par le radotage-jusqu'au-plafond, pas par du code plus long).

→ Conclusion transverse : **le modèle apprend à terminer (dans les deux envs), pas à raisonner/coder mieux, et son raisonnement ne s'allonge pas.**

---

## 2. Le mécanisme R1 vs nous

Le RL (GRPO) ne récompense **jamais la longueur directement**. La reward est **outcome** (correct/faux). L'avantage GRPO d'un rollout = `reward − moyenne du groupe`. Ce qui est renforcé = les rollouts à *haute reward*.

**R1 / gros base (V3-Base, 671B) :** sur un problème dur, **raisonner plus long ⇒ plus de chances de résoudre** (backtracking, vérification, exploration — capacités que le gros base possède déjà en latent). Donc dans un groupe, **les rollouts corrects SONT les longs** → avantage positif sur les longs → **le gradient de correction tire mécaniquement vers la longueur.** La longueur est récompensée *indirectement, via la corrélation longueur ↔ correct.* Le CoT grossit tout seul (« aha moments » émergents).

**Notre 2B :** raisonner plus long ⇒ **radotage**, pas plus de correct. **Aucune corrélation longueur ↔ correct** → le gradient ne tire pas vers la longueur. Pire, à correction égale, GRPO favorise légèrement le **court** (plus de masse de probabilité) → poussée vers le court.

---

## 3. La preuve empirique (dans nos propres chiffres)

Dans nos groupes de rollouts, les réponses **correctes ne sont PAS plus longues** que les fausses :

| env | longueur médiane correct | longueur médiane faux |
|---|---|---|
| Math | 2064 | 2067 (plat) |
| Code | 1020 | 1051 (correct légèrement **plus court**) |

→ **Il n'y a littéralement aucun signal « plus long ⇒ correct » que GRPO pourrait suivre.** Chez R1, le correct était le long ; chez nous, longueur et correction sont décorrélées.

---

## 4. La cause racine : le RL ÉLICITE la capacité latente, il ne la CRÉE pas

La corrélation « long ⇒ correct » n'existe **que si le base peut utiliser le raisonnement long productivement.**

- **671B** : capacité latente énorme → le RL la révèle et l'affûte.
- **2B** : peu de capacité latente → quand il raisonne long, il radote → rien à révéler.

**DeepSeek l'a montré eux-mêmes** : leurs modèles 1.5B–32B (`R1-Distill-*`) ne sont **pas** faits par RL — ce sont des **distillations** (SFT sur des traces générées par R1). Ils ont testé le RL GRPO directement sur Qwen-32B : **la distillation gagne largement.** Conclusion du papier : les petits modèles via large-scale RL n'atteignent même pas la performance de la distillation.

→ **On fait exactement le pire cas de leur étude** : RL pur (style R1-Zero, sans cold-start) sur un **2B**. Notre résultat (terminaison élicitée, capacité + longueur plates) colle parfaitement.

---

## 5. Est-ce à cause de BFT ? — Non (preuve logique)

BFT (la terminaison forcée) ne s'applique **qu'au math**. Le **code n'a aucun BFT** — et pourtant la longueur du code est plate elle aussi. **Si BFT était la cause, le code aurait dû grossir. Il n'a pas.** Donc BFT n'est pas la cause de fond.

BFT **est** quand même un vrai frein *math-spécifique* : il cape le thinking à 2048 (impossible de raisonner plus long) et force la réponse (découple la reward du raisonnement). À retirer quand le modèle termine seul — mais **n'attends pas** que ça déclenche une croissance type R1 : le mur reste le 2B.

---

## 6. Implications pour le design

1. **Le RL seul ne fabriquera pas la capacité sur un 2B.** Il continuera d'affiner la terminaison + les motifs de surface (overfit à la distribution d'entraînement) et de plafonner sur la compétence.
2. **La recette qui marche (celle de DeepSeek pour les petits modèles) : distillation / cold-start SFT d'abord** — injecter des traces de raisonnement de haute qualité (d'un modèle fort) — **puis** RL pour affûter. La capacité doit exister *avant* que le RL puisse l'allonger.
3. **Nettoyer la reward** (grader OCI : retirer les tests string-exact impossibles/fragiles ; retirer BFT quand la terminaison est acquise) pour que le RL d'affûtage soit honnête.
4. Réaliste : même bien distillé, un 2B a un plafond ; pour viser vraiment haut, un base plus gros.

**Message d'une ligne :** *le RL a cueilli le fruit facile (la terminaison) ; derrière il n'y a pas de gradient facile vers la compétence sur un 2B — il faut l'injecter par distillation, le RL ne la crée pas.*
