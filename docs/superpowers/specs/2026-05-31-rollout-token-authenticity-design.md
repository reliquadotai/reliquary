# Rollout token authenticity — détection d'injection de tokens

Date : 2026-05-31
Auteurs : investigation Claude + Romain

## Résumé

Des mineurs gagnants fabriquent leurs groupes de 8 rollouts en **injectant des tokens**
(typiquement le chiffre d'un résultat arithmétique) pour façonner artificiellement le
vecteur de récompense à `11111100` (6 corrects / 2 faux, σ=0.433), seul moyen de rester
dans la zone GRPO tout en maximisant le nombre de corrects. Les filtres actuels ne voient
pas l'injection. Ce spec ajoute un check d'**authenticité par token** : toute position où
le modèle était quasi-certain d'un autre token (proba du token émis effondrée) trahit une
édition post-hoc, et la soumission est rejetée.

## Problème

### L'exploit (confirmé)
- Le filtre de zone (`is_in_zone`, verifier.py:481 ; `SIGMA_MIN=0.43`, constants.py:234)
  rejette toute soumission dont l'écart-type des récompenses < 0.43. Pour 8 récompenses
  binaires, ça plafonne le nombre de corrects à **k≤6** (k=7 → σ=0.331 rejeté ; k=8 → σ=0).
- Pour maximiser les corrects en restant en zone, les mineurs visent **pile 6/2** sur
  *chaque* prompt. Mesuré : `reward_vector=11111100` sur **429/450** soumissions, σ=0.433
  sur **448/459**, et les 4 hotkeys dominants captent **~82 %** de l'émission avec ce
  vecteur à ~100 %.
- Le taux de réussite naturel du modèle varie pourtant de **12 % à 100 %** selon le prompt
  (mesuré en génération libre). Obtenir 6/8 indépendamment de la difficulté est
  mécaniquement impossible en sampling honnête → les groupes sont **construits**.

### Le mécanisme d'injection (preuve checkpoint-exacte)
Replay teacher-forced avec le checkpoint exact de la window (`R0mAI/reliquary-sn-v23`
@ `306c4af8`, = `claimed_checkpoint_hash`) : à la position du chiffre de réponse, le
modèle assigne **p≈1.000 au bon chiffre** et **1e-13 à 1e-26 au chiffre écrit**. À T=0.9
un token de proba 1e-20 n'est jamais échantillonné → le chiffre a été **forcé**. Les
jumeaux faux (42/40, 12/15) = une base correcte dupliquée, chiffre flippé en ±1/±2. Sur
les prompts durs (ex. gt=30, réussite modèle 25 %), même les rollouts **corrects** portent
des injections — ils fabriquent aussi les bonnes réponses.

### Pourquoi les filtres actuels ratent
- `evaluate_boxed_answer_probability` (verifier.py:668, seuil `BOXED_ANSWER_MIN_PROB=1e-3`,
  constants.py:515) ne regarde **que les tokens dans le dernier `\boxed{...}`**. L'injection
  est **16 à 67 tokens en amont** (à l'étape arithmétique) ; la box recopie le nombre déjà
  écrit → dedans il est auto-cohérent (proba ~1.0) → **PASS**. Mesuré : 16/16 faux ont
  `inside box? False`.
- `lp_dev` (logprob challenge) compare *claimed* vs *recomputed* ; le mineur reporte les
  vrais logprobs (même très bas) de ses tokens édités → `dev≈0`. Il attrape le mensonge sur
  les logprobs, pas l'injection.
- `dist_q10` : un seul token outlier ne bouge pas le 10ᵉ percentile.
- GRAIL (`verify_commitment_proofs`, verifier.py:238) prouve « ces hidden states sont
  cohérents avec le forward **sur ces tokens** » — pas « chaque token a été échantillonné ».
  Le mineur relance un forward sur la séquence éditée et commit des hidden states honnêtes.

## Calibration (faux positifs)

Génération honnête **vLLM** (stack mineur) puis vérif **HF forward** (stack validateur),
checkpoint exact, params protocole (T=0.9, top_p=1, top_k=0), **800 complétions** :

| population | tokens | min proba honnête |
|---|---|---|
| chiffres | 108 633 | **1.27e-3** |
| tous tokens | 550 411 | **3.5e-7** |

Faux positifs de la règle `p_chosen < seuil ET p_argmax ≥ 0.99` :

| seuil | tous tokens | chiffres seuls |
|---|---|---|
| 1e-3 | 44 | 0 |
| 1e-4 | 5 | 0 |
| 1e-6 | 1 (`}}`, artefact BPE) | 0 |
| **1e-8** | **0** | **0** |

Ablation : `argmax ≠ token` est **redondant** (impliqué par `p_chosen < seuil`). La
condition utile est `p_argmax ≥ 0.99`, qui écarte les positions où le modèle hésitait
légitimement. À 1e-8 aucun token honnête n'est concerné, donc 0 FP même sans la condition.

**Marge** : plancher honnête (tous tokens) 3.5e-7 ; plafond d'injection ≤1e-13 ; seuil
retenu **1e-8** → 0 FP mesuré, 5 ordres de marge au-dessus des injections.

## Design

### 1. Forward GPU : émettre l'argmax (`verifier.py`)
Étendre `_gpu_completion_chosen_probs` (verifier.py:~404). Sur le tenseur `probs`
`[n_steps, vocab]` déjà calculé, ajouter `amax_prob, amax_id = probs.max(dim=-1)` et
renvoyer `(chosen, argmax_ids, argmax_probs)`. Aucun forward ni transfert `[seq,vocab]`
supplémentaire. Porter `completion_argmax_ids` et `completion_argmax_probs` sur
`ProofResult` (verifier.py:~40) à côté de `completion_chosen_probs`.

### 2. Check `evaluate_token_authenticity` (`verifier.py`)
Signature alignée sur `evaluate_boxed_answer_probability`
`(tokens, prompt_length, completion_length, proof, tokenizer, *, threshold=1e-8, argmax_conf=0.99)`.
Parcourt **toutes** les positions de complétion ; flag la première position où
`p_chosen < threshold ET p_argmax ≥ argmax_conf`. Retourne `(ok, metrics)` avec la position
fautive, le token émis, l'argmax et les deux probas (télémétrie). Indexation alignée sur
`completion_chosen_probs` (même hypothèse que le filtre boxed). `ok=True` si pas de probas.

### 3. Câblage + enforcement (`batcher.py`, `server.py`)
Appeler le check là où `evaluate_boxed_answer_probability` est invoqué. Nouveau
`RejectReason.TOKEN_TAMPERED`. **Enforcement par soumission** : si **un seul** des rollouts
du groupe est flaggé, rejeter **toute la soumission** (le groupe est frauduleux). Couvre
les faux fabriqués **et** les corrects fabriqués.

### 4. Déploiement en deux temps
Constante `TOKEN_AUTH_ENFORCE` (défaut `False`).
- **Shadow** : calculer + logguer chaque flag (hotkey, window, prompt_idx, pos, token,
  argmax, p_chosen, p_argmax) sans rejeter. Confirmer sur 1-2 jours : 0 FP sur les honnêtes
  et flag effectif des hotkeys fabricants. Le shadow recalibre aussi le seuil sur le trafic
  prod réel.
- **Enforce** : bascule `True` = rejet dur.

### 5. Env code (exigence liée, hors implémentation env)
La récompense de l'env code doit être **recalculée par le validateur** (ré-exécution du
code soumis), jamais reprise du mineur. Couplé au check d'authenticité : éditer un token
ne peut **ni fausser une valeur de reward** (ré-exécutée) **ni fabriquer un faux échantillon**
(détecté). À vérifier à l'arrivée de l'env ; si absent, prérequis bloquant.

## Tests
- **Rejet** : rejouer les 16 rollouts faux de R2 (archive window 10547-10549) → flaggés.
- **0 FP** : les 800 complétions honnêtes vLLM → toutes acceptées.
- **Non-régression** : un tamper synthétique *dans* la box → toujours attrapé par l'ancien
  filtre (les deux checks coexistent).
- **Enforcement** : une soumission avec ≥1 rollout flaggé → `TOKEN_TAMPERED` sur tout le
  groupe.
- **Forward** : `_gpu_completion_chosen_probs` étendu renvoie des `argmax_*` alignés et
  cohérents avec un calcul CPU de référence.

## Hors périmètre / notes
- **L'incitation de fond reste** : la zone σ≥0.43 interdit de soumettre 8/8, ce qui pousse
  à fabriquer une variance. Ce spec tue la *fabrication par édition* ; repenser la zone
  (ne pas plafonner la justesse) est un sujet séparé.
- **Garantie** : un seuil de proba est statistique, pas une preuve cryptographique « zéro
  édition ». Il attrape toute édition à position confiante (= celles qui flippent une
  reward). Une garantie absolue demanderait un sampling déterministe vérifiable, infaisable
  cross-stack (drift vLLM↔HF).
- L'incohérence d'encodage prompt (brut archivé vs templaté canonique, branche
  `feat/chat-template-prompt-encoding`) crée du bruit bas-proba **non-chiffre** ; le seuil
  1e-8 et le ciblage sur positions confiantes l'absorbent. Fiabiliser l'encodage reste
  bénéfique en parallèle.
