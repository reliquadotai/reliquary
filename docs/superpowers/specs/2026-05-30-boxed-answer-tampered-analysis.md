# Analyse du filtre `boxed_answer_tampered` — robustesse & faux positifs

Date : 2026-05-30
Auteur : investigation Claude + Romain

## Problème signalé
Le filtre `boxed_answer_tampered` rejetterait des miners honnêtes (faux positifs),
surtout sur les réponses boxées longues. Question : version plus robuste (idée softmax),
validable par des tests.

## Filtre actuel
`evaluate_boxed_answer_probability` (reliquary/validator/verifier.py:669) :
- recalcule pendant la vérif `completion_chosen_probs` (softmax à T_PROTO=0.9, token choisi)
  pour chaque token du dernier `\boxed{...}` ;
- prend le **min** sur la zone boxed ;
- rejette si `min < BOXED_ANSWER_MIN_PROB = 1e-3` (constants.py:515).
Défaut structurel : statistique pire-cas (min), seuil absolu, aucune normalisation,
aucune prise en compte de la distribution complète (les logits sont pourtant sur GPU).

## Données
- Archive R2 `reliquary/dataset/window-*.json.gz` (61 windows OMI).
  **Ne contient que les rollouts ACCEPTÉS** (tokens+gt) ; les rejetés n'ont que des
  métadonnées (reason, sketch_diff_max, lp_dev_max, dist_q10_min) — **pas de tokens**.
- Rejets réels : **24 `boxed_answer_tampered` / 61 windows** (rejet #1 hors bad_termination).
- Ground-truths longs = LaTeX : `\begin{pmatrix}...`, `\frac{5x^2+3x+7}{x^2+x-2}`.
- Modèle : `R0mAI/reliquary-sn-v23` (Qwen3-4B fine-tuné), H100.
- Régénérations honnêtes : T=0.9, top_p=1, top_k=0 (params protocole), forward HF.
- Positifs tampered : swap du contenu du `\boxed{}` (attaque réelle), re-tokenisé.

## Résultats MESURÉS

### Réponses courtes (1159 paires)
| règle | FP honnête | recall tampering |
|---|---|---|
| actuel (min<1e-3) | 0.00 % | 100.00 % |

Séparation parfaite (tampered min≈0.000 ; honnête min=1.0).

### Dégradation du min honnête avec la longueur de box (stack identique)
| tokens box | p10(min) | min observé |
|---|---|---|
| 1–2 | 1.00 | 1.00 |
| 5–8 | 0.84 | 0.30 |
| 17+ | 0.16 | 0.16 |

### Réponses longues (470 paires, box jusqu'à 51 tokens)
- min honnête : jamais < 0.01 (min absolu 0.0497, p01 0.245)
- actuel (min<1e-3) : **0.00 % FP, 100 % recall**

## Conclusion
- Faux positif **NON reproduit en stack identique**, même sur réponses longues. La
  dégradation du min existe mais ne franchit pas 1e-3 quand génération=vérif.
- Seul facteur restant plausible : **mismatch stack miner (vLLM) ↔ validateur (HF
  forward)**, seul capable d'expliquer la chute de ~2 ordres de grandeur. Test monté
  (gen vLLM + capture logprob → revérif HF) mais non terminé (recyclage du conteneur
  GPU + incompat vLLM 0.11 / transformers 5.9 ; downgrade 4.57.1 requis).

## Piste de correctif (à valider sur un VRAI faux positif)
Ajouter une condition de confiance à la règle de rejet :
rejeter seulement si un token boxed a `p_chosen < thr` **ET** `p_argmax > C`
(le modèle voulait fortement un autre token). Sur les positifs synthétiques, le token
swappé a `p_argmax=1.0, entropy=0.00` → ce critère préserve 100 % de recall tout en
ignorant les effondrements honnêtes de queue d'échantillonnage. NON encore validé
contre un FP réel.

## Prochaines étapes possibles
1. Box GPU stable → finir le test vLLM→HF (reproduire le FP, mesurer FP/recall du correctif).
2. Capturer un vrai rejet `boxed_answer_tampered` AVEC ses tokens (instrumenter le
   validateur : l'archive ne les conserve pas).
3. Shadow-log de la distribution réelle de `min_prob` en prod avant tout changement.

## Artefacts (locaux)
- `/tmp/honest_rollouts.jsonl` (3664 rollouts OMI acceptés extraits de R2)
- `/tmp/long60.jsonl` (60 prompts à réponse longue)
- `/tmp/tamper_long.json` (470 paires, stats par token boxed)
- scripts : `/tmp/regen.py`, `/tmp/tamper_eval.py`, `/tmp/diag_boxed.py`,
  `/tmp/gen_vllm.py`, `/tmp/cmp_vllm_hf.py`
