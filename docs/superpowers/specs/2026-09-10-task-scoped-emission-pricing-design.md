# Tarification de l'émission et contrats par tâche

**Dernière révision** : 2026-09-11
**Statut** : design validé. **Phases 0 et 1 de la V1 implémentées** sur cette branche : le prix est calculé et publié dans l'archive, **jamais appliqué**. Tout le reste est à faire.

> **Ce qui a changé depuis la version du 2026-09-10**
> - L'horloge est le **round drand**, plus la hauteur de bloc (§4, §5).
> - Le dénominateur de `r` est la **durée de la fenêtre**. Mesurer chaque étage devient optionnel (§5).
> - Le snap **escalade depuis le prix courant** (§4).
> - Ajout du **contrat par tâche** et du **miner piloté par le contrat** (§7). C'est la pièce qui permet de changer les paramètres d'une tâche sans casser les miners. La version précédente n'en parlait pas.
> - Ajout de l'**accès aux informations** (§8) et du **flux opérateur** (§9).
> - Phases réordonnées (§15) : les contrats passent avant le multi-tâches.

### Où est le code

| Fichier | Contenu |
|---|---|
| `reliquary/validator/emission_price.py` | contrôleur pur, mesure de l'offre, lecture d'archive, paramètres de départ |
| `reliquary/validator/service.py` | `_window_price_signal`, `_price_target_for`, `_advance_price_shadow` |
| `tests/unit/test_emission_price_*.py`, `test_archive_carries_price_*.py`, `test_window_price_signal_extraction.py` | 53 tests, écrits pour se lire comme la spec |

---

## 1. Problème et objectifs

Deux problèmes, un seul design.

**L'émission est une dépense décorrélée de ce qu'elle achète.** Le pool est câblé à `1.0` par fenêtre (`service.py`, `window_pool=1.0`) : la totalité de la part miner est versée, quelle que soit l'offre réelle, le besoin réel du trainer ou la vitesse de remplissage de la fenêtre.

Estimation d'ordre de grandeur, hypothèses visibles :

| | |
|---|---|
| Fenêtres/jour (512 preuves au goulot de 11/min) | ~31 |
| Séquences prouvées/jour | ~254 k |
| Tokens générés/jour, prouvés | ~127 M |
| Avec sur-génération ×10 | ~1,3 Md |
| Coût marchand (4B, vLLM H100 ~4 k tok/s, ~2 $/h) | **~180 $/j** |
| Émission versée | **~6 000 $/j** |

L'écart apparent est de l'ordre de 30×. Il n'est **pas** entièrement récupérable. Un miner peut miner sur n'importe quel subnet : son prix n'est pas son coût de calcul mais ce qu'il gagnerait ailleurs. Et une part de l'écart a été *dissipée* dans la course à la latence (sélection 62/24/5 % par tercile d'arrivée) plutôt que capturée comme marge. Le passage au fill-closed a largement fermé ce canal, ce qui rend une partie de l'écart capturable — **on ne sait pas laquelle**.

**Chaque changement de paramètre casse tout le parc.** Le miner refuse tout contrat de génération différent de celui compilé dans son package (§2). Changer une température, un template, un budget de tokens, ou ajouter une tâche, impose une release miner et une mise à jour de tout le monde. Itérer vite est impossible.

> **Objectifs**
> 1. Minimiser l'émission versée sous contrainte que la tâche reste servie, et **détruire le reste**. Le montant n'est pas deviné : il est mesuré.
> 2. Créer une tâche ou changer ses paramètres **sans release miner**, tant qu'elle reste d'un type déjà supporté.

Le burn est une destruction réelle : `UID_BURN` non défini vaut « le uid du validateur lui-même » (`constants.py`), et **une règle côté chaîne** — hors de ce repo, donc non vérifiable ici — veut qu'un validateur ne perçoive pas d'incitatif miner : la masse allouée là disparaît. Il n'y a donc pas de conflit d'intérêt entre « qui calcule le prix » et « qui reçoit le résidu ».

**Les deux objectifs se renforcent.** Avec un prix découvert et un burn du résidu, chaque tâche prend ce dont elle a besoin : il n'y a plus de split à négocier, donc ajouter une tâche ne demande aucune décision économique. Avec des contrats par tâche, l'ajouter ne demande aucune release. Ensemble, une nouvelle tâche devient une simple publication.

---

## 2. Invariants découverts

Vérifiés dans le code (base `68971b02`). Ils contraignent tout ce qui suit.

### Émission et rejeu

**Une seule identité on-chain.** Le signer applique une politique `(netuid, hotkey)` unique et n'expose que trois opérations typées (`/v1/checkpoints/sign`, `/v1/weights/set`, `/v1/axon/serve`). Un seul vecteur de poids existe ; tout split doit être résolu avant `set_weights`.

**`_replay_ema` ne recalcule rien.** Il lit `record.get("rewards_by_hotkey", {})` et rejoue tel quel. Toute mise à l'échelle faite par le validateur qui écrit l'archive est reproduite à l'identique par n'importe quel lecteur, ancienne ou nouvelle image. *C'est ce qui rend la V1 déployable sans coordination.*

**`archive_schema_version` est écrit et jamais vérifié.** Aucun consommateur ne le lit. Ajouter des champs à l'archive est gratuit.

**Le burn absorbe le résidu, mais jamais au-delà.** `burn_weight = max(0.0, 1.0 - registered_total)`. Si la somme dépasse 1, le burn disparaît silencieusement et la chaîne renormalise. **`Σ ≤ 1` n'est pas négociable.**

**L'EMA compte les archives, pas le temps.** `EMA_ALPHA = 2/(72+1)`, 216 archives rejouées. L'état stationnaire d'un miner vaut `ρ × pool`, où `ρ` est la part d'archives produite par sa tâche. En mono-tâche `ρ = 1` et la durée des fenêtres ne compte pas ; en multi-tâches, **le débit d'archives décide du split**.

**La constante d'EMA a dérivé.** Son commentaire dit « 72 windows ≈ ~6 hours » — calibré pour des fenêtres de ~5 min. Sous fill-closed (~46 min), la constante de temps réelle est de ~28 h et l'historique couvre ~7 jours. Personne ne l'a décidé.

### Fenêtre et horloge

**Le numéro de fenêtre est un compteur, pas un bloc.** `self._candidate_window_n = self._window_n + 1`. Sous fill-closed la fenêtre ferme au remplissage, pas à l'horloge.

**Une fenêtre fill-closed qui ne se remplit pas ne se ferme pas.** `FILL_CLOSED_TARGET_GROUPS_PER_ENV` = « *Proven groups that close one environment* ». Pas de dégradation douce : la pénurie arrête le trainer.

**Le temps de fermeture mesure notre propre plan de preuve.** La fenêtre ferme sur 256 groupes **prouvés** par environnement, et le goulot mesuré est de 11 preuves/min pour 25,6 nécessaires. Doubler les miners ne raccourcirait pas la fenêtre. *Mesurer au mauvais endroit tue le capteur ; voir §5.*

**La boucle de fenêtre ne fait aucun appel chaîne, mais elle connaît la balise.** `get_current_block` n'est jamais appelé par le service. En revanche `window_open_drand_round` est posé depuis la balise drand à l'ouverture, et `_seal_trigger_round` au scellement.

**L'index par prompt n'est jamais purgé.** `_submissions_per_prompt` est complété à chaque admission et jamais vidé. Chaque `PendingSubmission` porte son `drand_round`, validé contre la balise à la réception.

### Contrat de génération

**Le miner compare le contrat, il ne l'applique pas.** `miner/engine.py:78-81` :

```python
state.generation_profile_id == ACTIVE_PROTOCOL_PROFILE.profile_id
and state.generation_contract == to_generation_contract(ACTIVE_PROTOCOL_PROFILE)
```

Le miner reçoit le contrat du validateur mais ne s'en sert que pour vérifier qu'il est **identique** au sien. Sinon : `protocol_mismatch`, il ne soumet rien. `RELIQUARY_PROTOCOL_PROFILE` ne choisit que parmi les profils déjà écrits dans `PROFILES`. Échantillonnage, template, budget de tokens et modèle sont figés dans le package du miner.

**Le validateur fait pareil.** `validator/server.py:280` rejette une soumission dont `generation_profile_id` diffère de `PROTOCOL_PROFILE_ID`.

**Le contrat existe déjà comme objet détaché, avec un digest.** `ProtocolProfile.to_generation_contract()` produit un objet JSON natif : modèle, encodage de prompt, et par environnement le budget de tokens, le format de réponse, le BFT, le template signé, l'ABI d'épisode, l'identité d'environnement. Le plan de preuve distant l'identifie déjà par `generation_contract_sha256`.

**La révision du checkpoint est déjà une donnée.** `MinerState.checkpoint_revision` change au fil du run sans release miner. C'est le précédent de tout le §7 : on traite le reste du contrat de la même façon.

**Le discriminant de tâche est déjà sur le wire.** `BatchSubmissionRequest.generation_profile_id`, défaut `""`, `max_length=64` — exactement la longueur d'un sha256 en hexadécimal.

**Les environnements externes existent déjà.** `EnvironmentSpec` accepte `external_distribution` et `environment_manifest_sha256` : le code d'un environnement peut vivre dans un paquet séparé, identifié par digest.

### Wire et découverte

**`MinerState` est `extra="forbid"`.** Ajouter un champ à une réponse existante casse tout miner qui garde l'ancien modèle. Les extensions passent par des **paramètres de query** et des **routes neuves**. Le repo l'a déjà fait : `/miner-state` est né comme *« additive endpoint »* à côté de `/state`, et `RuntimeContract` comme *« capability response served separately from strict legacy /state »*.

**Les releases se négocient par capacité, pas par version.** `protocol/release_contract.py` décrit ce qu'un pair implémente *« without using an integer version as feature detection »*, avec `canonical_json_bytes` pour une sérialisation déterministe.

**Le signer refuse de signer n'importe quoi.** `CheckpointSignRequest` est typée `(checkpoint_n, repo_id, revision)`, pas des octets arbitraires. Propriété de sécurité à préserver.

**L'espace de noms R2 est plat.** `reliquary/dataset/window-{n}.json.gz`, et `list_all_window_keys` est fixé sur ce préfixe.

**Les miners découvrent le validateur par le premier axon.** `miner/submitter.py` : *« Picks the first uid with validator_permit=True »*. Un second hotkey validateur détournerait silencieusement une partie du parc.

**Les soumissions sont grosses.** `MAX_SUBMISSION_PAYLOAD_BYTES = 64 MB`.

---

## 3. Unité de compte

### V1 — une tâche

```
window_pool = price
```

C'est tout. Calcul du score, sélection, ordre d'arrivée, tie-break, partage par slot et burn des slots non remplis : **rien ne change**. `price` remplace le littéral `1.0` au seul point d'injection existant.

En mono-tâche, l'état stationnaire de l'EMA vaut exactement le pool par archive, **indépendamment de la durée des fenêtres**. Un facteur de durée ici ne corrigerait rien et créerait une dépendance qui n'existe pas aujourd'hui — d'autant plus gênante que les fenêtres fill-closed n'ont pas une durée constante et vont raccourcir.

### Multi-tâches — plus tard, et par paire

Avec N tâches, `ρ_i` multiplie le pool de la tâche *i* : deux tâches écrivant `pool = 1.0`, la seconde deux fois plus rapide, se partagent l'émission 32 / 68 sans que personne ne l'ait décidé (vérifié par simulation sur une copie fidèle de `_replay_ema`).

La correction est **une paire indissociable** :

1. **écriture** — la fenêtre paie `price_i × Δrounds_i / ROUNDS_RÉFÉRENCE` ;
2. **lecture** — la décroissance de l'EMA suit le **temps écoulé**, pas le nombre d'archives : `ema *= (1 − α) ** (Δrounds / ROUNDS_RÉFÉRENCE)`.

Seule, (1) corrige le *ratio* entre tâches mais laisse dériver le *niveau* total. Ensemble, l'état stationnaire vaut `price_i` pour chaque tâche et `Σ price_i ≤ 1` redevient structurel. Cette paire exige une mise à jour de tous les lecteurs (§13) : elle appartient à la V3.

---

## 4. Le contrôleur *(implémenté)*

### Ce qu'il suit

```
r = t_collect / t_incompressible
```

`t_collect` est le moment où l'offre a atteint la cible de la fenêtre. `t_incompressible` est le temps que la fenêtre aurait pris avec une collecte instantanée (§5 pour les deux).

`r < 1` : la collecte est masquée par les autres étages. La vitesse de l'offre achète quelque chose qu'on ne peut pas consommer — **on paie du vide**. `r` proche de 1 : la collecte est sur le chemin critique.

La cible est **notre propre performance**. Quand le plan de preuve s'améliore, la fenêtre raccourcit, la cible avec, et le prix suit sans qu'on retouche un chiffre.

### Les trois régimes

```
r̄ = médiane de r sur les median_rounds derniers rounds
    (fenêtre non remplie : compte comme +∞ ; fenêtre remplie mais non mesurée : exclue)

  fenêtre non remplie    →  price = max(price, last_good) × snap                  [snap]
  r̄ < deadband           →  price ×= decay ** (Δrounds / rounds_per_step)        [descente]
  sinon                  →  hold                                                 [bande morte]

  fenêtre remplie        →  last_good = price
  price = clamp(price, floor, cap)
```

**Le snap escalade depuis le prix courant.** Une fenêtre non remplie arrête le trainer : on ne marche pas, on saute. Et on repart du prix *courant*, pas de `last_good` : épingler le snap à `last_good × snap` le laisserait bloqué au même prix pour toujours si le premier saut ne suffit pas à ramener l'offre. Deux pénuries d'affilée donnent `× 1,20` puis `× 1,44`. C'est aussi ce qui autorise une descente rapide.

**L'exigence de preuve est asymétrique parce que le coût de l'erreur l'est.** On descend sur le signal **lissé** : dépenser moins doit se mériter. On remonte sur le signal **instantané** : restaurer la liveness ne se négocie pas.

**La bande morte** supprime le frétillement autour de la cible, principale source d'oscillation quand l'offre est élastique et que tous les miners lisent le même prix au même moment.

**Une fenêtre remplie mais non mesurée est du silence, pas une pénurie.** Confondre les deux ferait monter le prix à chaque raté d'instrumentation — et le snap est le seul régime qui ne demande aucune confirmation.

### Toutes les constantes en rounds

> **Aucune constante du contrôleur ne se compte en fenêtres.**

`EMA_ALPHA` montre pourquoi : la constante n'a jamais bougé, c'est son unité qui a changé sous elle. Un contrôleur réglé en « −1 % par fenêtre » accélérerait silencieusement le jour où les fenêtres passent de 46 à 8 minutes. Décroissance, médiane et horizon d'affichage sont donc exprimés en rounds drand.

### État borné

`advance(state, recent, params)` prend l'état de la fenêtre précédente (`price`, `last_good`) et les fenêtres récentes, pas l'historique complet. `_replay_ema` ne lit qu'une tranche bornée d'archives : un prix qui exigerait de remonter au début du run mettrait un lecteur et l'écrivain sur deux nombres différents.

### Cadence et affichage

La boucle que le contrôleur ferme est :

```
prix affiché → décision du miner → t_collect → contrôleur      ~1 fenêtre
émission réalisée ← chaîne ← EMA                              en aval, PAS dans la boucle
```

Le contrôleur ne regarde jamais l'émission versée. Le délai de boucle est donc l'allumage d'un miner (~4 min sur instance spot) plus une fenêtre, pas les 28 h de l'EMA. **La marge de sécurité vient du lissage, pas du ralentissement** : ralentir jette de l'information, lisser la garde.

Le prix est **affiché à l'avance**, et le prix qui s'applique à une fenêtre est **celui en vigueur à son ouverture**. Un miner doit connaître sa marge *avant* de payer son instance. Sinon il ne peut réagir qu'à son revenu réalisé, le retard de l'EMA entre dans la boucle et le système oscille.

### Paramètres de départ

`PRODUCTION_PRICE_PARAMS`, dans `emission_price.py` :

| Paramètre | Valeur | Lecture |
|---|---|---|
| `start` | 1.0 | le pool actuel : armer ne change rien à la première fenêtre |
| `decay` / `rounds_per_step` | 0.99 / 1000 | −1 % toutes les ~50 min, soit ~−25 %/jour |
| `deadband` | 0.80 | |
| `snap` | 1.20 | |
| `floor` | 0.05 | garde de liveness, pas une opinion économique |
| `cap` | 1.0 | |
| `median_rounds` | 4800 | ~4 h de confirmation |

**Aucune valeur n'est arrêtée.** La fenêtre de médiane et le plancher de liveness se *dérivent* du délai de boucle et doivent être recalculés s'il change. `decay`, la bande morte, le snap et le plancher se calibrent sur la courbe observée en shadow, avant tout armement.

Ordres de grandeur avec ces valeurs : 1,0 → 0,5 en ~2 jours, → 0,1 en ~7 jours, → 0,05 en ~10 jours. Avec la bande morte et la médiane, compter 12 à 15 jours.

---

## 5. La mesure *(implémentée)*

### L'horloge : le round drand

Quicknet, un round toutes les 3 secondes. Pas la hauteur de bloc : la boucle de fenêtre ne fait aucun appel chaîne, alors que `window_open_drand_round` est déjà posé depuis la balise et que tout le scellement raisonne en rounds. Les rounds sont trois fois plus fins et ne coûtent rien de plus. Le nom compte : appeler un round un « bloc » serait exactement la confusion d'unité qui a fait dériver `EMA_ALPHA`.

### Trois champs, tout ou rien

| Champ d'archive | Source |
|---|---|
| `window_open_round` | `window_open_drand_round`, posé depuis la balise |
| `window_close_round` | `_seal_trigger_round` |
| `collect_ready_round` | round où l'offre a atteint la cible — `null` si elle ne l'a jamais atteinte |

Les trois partent ensemble ou pas du tout. Un enregistrement avec les bornes de fenêtre mais sans `collect_ready_round` serait lu comme une **pénurie** — le seul régime qui fait monter le prix sans confirmation — alors que le validateur n'a simplement pas pu mesurer. Le silence doit ressembler au silence.

- **Trio absent** : fenêtre non mesurée (toutes les archives antérieures, ou batcher incomplet). Le contrôleur l'ignore.
- **`collect_ready_round: null` dans un trio complet** : mesurée, et elle ne s'est pas remplie. Vraie pénurie.
- **Fenêtre avortée** : ignorée, comme `_replay_ema` le fait déjà.

### `collect_ready_round` : sur l'ensemble admissible, par prompt distinct

C'est la différence entre un capteur vivant et un capteur mort. La fermeture de la fenêtre mesure notre plan de preuve (§2). Ce qui mesure l'offre, c'est l'**ensemble admissible** : les candidats qui ont passé tous les contrôles bon marché et auraient été prouvés s'il y avait eu la capacité.

1. Pour chaque prompt, on retient le round de son **premier** candidat admissible. La cible compte des *groupes*, un groupe = un prompt, et `MAX_SUBMISSIONS_PER_PROMPT` permet à dix candidats de viser le même : compter les soumissions signalerait une offre inutilisable.
2. On prend le N-ième de ces rounds, avec N = `FILL_CLOSED_TARGET_GROUPS_PER_ENV`. Pas le `batch_target` de l'archive : c'est la taille d'un batch d'entraînement, seize fois plus petite, qui déclarerait la fenêtre prête bien trop tôt.
3. L'environnement le plus lent décide. Math et code se remplissent à des rythmes très différents ; faire la moyenne donnerait une disponibilité qu'aucun des deux n'a eue.

Tout se calcule au scellement à partir d'état existant (`_submissions_per_prompt`, `drand_round`) : rien n'est ajouté au chemin d'admission.

### Le dénominateur : la durée de la fenêtre

Une fenêtre fill-closed se ferme quand la **dernière** contrainte est levée. Sa durée vaut donc `max(t_collect, t_étages)` — exactement le dénominateur voulu, gratuitement. C'était le plus gros morceau de travail prévu : il n'existe aucun chronométrage d'étage dans la boucle de fenêtre.

Coût : `r` est plafonné à 1. Il ne dit pas *de combien* la collecte a dépassé les étages. Le contrôleur n'en a pas besoin (il n'a pas de régime « monter parce que c'est lent » : la pénurie, c'est une fenêtre non remplie). Si des durées d'étage mesurées sont archivées (`training_rounds`, `validation_rounds`), elles priment, sans changement de schéma.

### La décision shadow

```jsonc
"emission_price_shadow": {
  "price": 0.97, "last_good": 0.97,
  "r": 0.21, "r_smoothed": 0.24,
  "regime": "descend",
  "applied": false
}
```

Publiée à chaque fenêtre mesurable, **jamais appliquée**. Un test le vérifie en comparant une fenêtre payée par un service dont le prix a déjà descendu avec la même fenêtre payée par un service neuf : les récompenses sont identiques.

---

## 6. Rejeu déterministe

Le prix n'est **pas déclaré, il est calculé**. C'est une fonction déterministe de la chaîne d'archives, comme l'EMA : n'importe qui relit les archives, rejoue la même récurrence sur les mêmes entrées et retombe sur le même chiffre.

1. **Aucune discrétion à l'exécution.** Rien n'est surchargeable par variable d'environnement. En V1, sans registre, les paramètres sont des constantes du module. Avec le registre (§10), ils vivent dans le registre signé : publiés avec préavis, chaînés et cités par chaque archive. Ce n'est pas une discrétion, c'est une publication vérifiable.
2. **Le prix ne va pas dans le registre signé.** Sinon il faudrait re-signer à chaque fenêtre, et on perdrait la vérification indépendante qui le rend crédible.
3. **Le `cap` reste dans le code.** C'est le paramètre qu'un attaquant ayant l'écriture sur le registre voudrait toucher. Le registre décide *comment* une tâche est payée ; la borne de ce qu'elle *peut* prendre est une constante versionnée.

**Trou connu, à combler avant d'armer.** En shadow, la marche du prix vit en mémoire : un redémarrage la remet à `start`. Sans conséquence tant que rien n'est appliqué ; une fois armé, **chaque redémarrage rendrait aux miners le pool entier**. L'état doit être amorcé depuis la dernière archive. `_load_archive_range`, qui fusionne déjà archives distantes et file locale, est le point d'accroche.

---

## 7. Contrat par tâche

### Le principe

> **Un contrat = tout ce qu'un miner doit savoir pour produire du travail valide sur une tâche.**

Il existe déjà (`to_generation_contract()`), mais il est **compilé et comparé** (§2). Le changement de fond est d'en faire une **donnée, appliquée** : le miner configure sa génération à partir du contrat qu'il reçoit, au lieu de le comparer à sa copie. Le validateur vérifie les soumissions contre ce même contrat, qu'il a publié et signé. La vérification reste exacte.

La révision du checkpoint fonctionne déjà ainsi (§2). Il s'agit de traiter le reste du profil pareil.

### Contenu

```jsonc
{
  "schema": "reliquary/task-contract/v1",
  "task_type": "rl_generation/v1",
  "model": {
    "model_id": "Qwen/Qwen3-4B-Base",
    "model_revision": "…",
    "checkpoint_source": "hf:aivolutionedge/reliquary-sn"   // RL : la révision courante reste un état live
  },
  "sampling": { "rollouts": 16, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "do_sample": true },
  "prompt_encoding": "…",
  "environments": {
    "openmathinstruct": {
      "environment_contract_id": "…",
      "environment_manifest_sha256": "…",
      "prompt_template": { "id": "…", "renderer": "dollar-substitution-v1", "sha256": "…" },
      "max_new_tokens": 4096,
      "answer_format": "boxed",
      "bft": null
    }
  },
  "requires": {
    "capabilities": ["generation.contract-driven/v1"],
    "max_context_tokens": 8192
  }
}
```

*(valeurs illustratives)*

Pour une tâche RL, le contrat fixe le modèle de base et la source des checkpoints ; la révision courante continue de circuler dans l'état live, comme aujourd'hui. Pour une tâche de génération de données sur checkpoint figé, la révision est fixée dans le contrat.

### Immuable, identifié par son contenu

```
contract_sha256 = sha256(canonical_json_bytes(contract))
```

Un contrat ne change jamais en place. **Changer un paramètre = publier un nouveau contrat, avec un nouveau digest**, et faire pointer la tâche dessus dans une nouvelle version du registre (§10).

**Règle de liaison** : une fenêtre est liée au contrat en vigueur **à son ouverture**, exactement comme le prix. Tout le travail d'une fenêtre est jugé avec un seul contrat ; un changement s'applique à la première fenêtre qui s'ouvre après le round d'activation. Aucun travail en cours n'est rejeté par une bascule.

**Liaison soumission ↔ contrat** : la soumission cite le digest. `generation_profile_id` fait déjà 64 caractères maximum, la longueur d'un sha256 hexadécimal, et il est inclus dans la signature de l'enveloppe. Il peut donc porter le digest sans changer le wire, pour les miners qui déclarent la capacité `generation.contract-driven/v1` ; les autres continuent d'envoyer l'identifiant de profil. *Changer le sens d'un champ existant est une décision à confirmer à l'implémentation.*

### Le miner piloté par le contrat

À la réception d'un contrat, le miner :

1. vérifie la signature du registre et le digest du contrat ;
2. vérifie qu'il a les **capacités** requises et que les **bornes** tiennent (contexte, mémoire GPU) ;
3. si oui, configure échantillonnage, rendu de prompt et budgets depuis le contrat ;
4. si non, **ne mine pas cette tâche** — il ne plante pas et continue sur les tâches qu'il sait servir.

Le contrat ne peut que **choisir parmi du code que le miner embarque déjà** : un renderer connu, un format de réponse connu, un ABI d'épisode connu. Il ne transporte jamais de code.

### Ce qui devient une donnée, ce qui reste une release

| Changement | Release miner ? |
|---|---|
| Paramètres économiques (prix, contrôleur) | **Non** — déjà vrai en V1 |
| Paramètres d'une tâche d'un type supporté : échantillonnage, rollouts, budget de tokens, template, tranche de dataset, modèle de base | **Non** — nouveau contrat, activé avec préavis |
| Nouvel environnement d'un type supporté, livré en paquet externe | **Non pour le protocole** — le miner installe le paquet s'il veut la tâche |
| Nouveau *type* de tâche, nouveau renderer, nouveau sampler, nouvel ABI d'épisode | **Oui**, mais sans casser : seuls les miners qui veulent la tâche mettent à jour |

### Côté validateur

Chaque validateur GPU sert une tâche. Au démarrage, il résout son contrat depuis le registre signé au lieu de `RELIQUARY_PROTOCOL_PROFILE` → `PROFILES`. Principale difficulté : beaucoup de constantes sont dérivées **à l'import** de `constants.py` à partir du profil actif (`M_ROLLOUTS`, `B_BATCH`, `T_PROTO`, `SIGMA_MIN`…). Le profil doit donc être construit depuis le contrat avant ces dérivations.

Deux étapes :
- **d'abord** : changer le contrat d'une tâche redémarre le validateur *de cette tâche* au round d'activation. Simple, n'affecte ni les autres tâches ni les miners — mais hérite de la fragilité actuelle des redémarrages (pas de reprise automatique) ;
- **ensuite** : bascule à chaud à l'ouverture de fenêtre, quand ces constantes ne sont plus globales.

### Le coût, franchement

**Une grosse mise à jour des miners, une seule fois.** Rendre le miner piloté par le contrat est précisément le changement qui oblige tout le monde à mettre à jour. On ne peut pas l'éviter — mais c'est la dernière fois pour les paramètres.

---

## 8. Toutes les informations au même endroit

### Une seule source de vérité

Le **registre signé** (§10) : tâches, digests de contrat, paramètres d'incitation, rounds d'activation. Tout le reste en dérive et peut être reconstruit à partir de lui et des archives.

### Points d'accès

**`GET /tasks`** — sur le front, servi depuis sa copie du registre signé plus l'état live. Supporte l'ETag comme `/miner-state`.

```jsonc
{
  "registry": { "sequence": 8, "sha256": "…", "signature": "…", "effective_from_round": 32104000 },
  "upcoming": { "sequence": 9, "effective_from_round": 32110000 },   // publié, pas encore actif
  "tasks": [{
    "task_id": "rl-math-code-v1",
    "task_type": "rl_generation/v1",
    "status": "active",
    "contract_sha256": "3f2a…",
    "contract_url": "/contracts/3f2a…",
    "requires": { "capabilities": ["generation.contract-driven/v1"], "max_context_tokens": 8192 },
    "price": { "posted": 0.73, "applied": false, "regime": "descend", "r_smoothed": 0.41,
               "valid_from_round": 32103120 },
    "window": { "window_n": 45702, "state": "open", "opened_round": 32103120 },
    "endpoints": { "state": "/t/rl-math-code-v1/miner-state", "submit": "/t/rl-math-code-v1/submit" }
  }]
}
```

**`GET /tasks/{task_id}`** — une tâche. **`GET /contracts/{sha256}`** — un contrat ; immuable, donc cacheable indéfiniment.

**Miroir R2** à côté des archives : `reliquary/tasks/registry/{sequence}.json`, `reliquary/tasks/registry/latest.json`, `reliquary/tasks/contracts/{sha256}.json`. Les lecteurs d'archives (nœuds weight-only) n'ont pas besoin de parler au validateur.

**Chaque archive cite** `task_id`, `registry_sequence` et `contract_sha256`. N'importe quel lecteur sait exactement ce qui était en vigueur pour chaque fenêtre.

### Propriétés

- **Le prix affiché est vérifiable** : `/tasks` le montre par commodité, mais il se recalcule depuis les archives (§6).
- **L'historique est auditable** : le registre est chaîné, donc « qu'est-ce qui a changé à la séquence 8 » est un diff entre deux fichiers signés.
- **La version à venir est visible avant d'être active** : un miner voit un changement de modèle ou de paramètres avant son activation, et peut s'y préparer.
- **Une page lisible** (tableau de bord) peut afficher `/tasks` tel quel. Elle ne fait qu'afficher : elle ne détient aucune vérité.
- **Compatibilité** : routes neuves uniquement, aucun champ ajouté à une réponse existante.

---

## 9. Travailler vite : le flux opérateur

```
1. éditer    tasks/rl-math-code-v1.json                    contrat + paramètres d'incitation

2. valider   reliquary task validate tasks/rl-math-code-v1.json
             schéma, bornes, type de tâche supporté, composants requis,
             mécanisme d'incitation connu, capacités cohérentes

3. publier   reliquary task publish tasks/rl-math-code-v1.json --from-round <R>
             JSON canonique, digests, signature par l'opération dédiée du signer,
             dépôt dans R2, séquence +1, chaînage

4. préavis   la séquence apparaît dans "upcoming" sur /tasks ;
             les miners pré-téléchargent si le modèle change

5. activer   au round R, la première fenêtre qui s'ouvre applique le nouveau contrat
```

**Retour arrière** : republier le digest précédent. Les contrats étant identifiés par leur contenu, le retour est exact et immédiat.

**Où est la sécurité** : éditer un fichier ne donne aucun pouvoir. Seule la signature par le signer (mTLS, PKI dédiée) rend une publication valide. La clé ne quitte jamais le signer.

**Préavis minimum** : un nombre de rounds constant dans le code, imposé par l'outil de publication *et* par les validateurs (qui refusent une version qui s'activerait trop tôt). L'outil prévient quand le modèle de base change, puisque c'est le seul changement qui demande un téléchargement aux miners.

---

## 10. Registre de tâches

> **Ajouter une *tâche* = une entrée de données. Ajouter un *type de tâche* = une release.**

```jsonc
{
  "schema": "reliquary/task-registry/v1",
  "sequence": 8,
  "previous_sha256": "…",              // chaînage : historique en ajout seul
  "effective_from_round": 32104000,    // activation différée, jamais immédiate
  "tasks": [{
    "task_id": "rl-math-code-v1",
    "status": "active",                // draft | active | draining | retired
    "contract": "3f2a…",               // sha256 du contrat (§7)
    "components": { "validator": "…", "trainer": "…" },
    "incentive": {
      "mechanism": "fill-rate-controller/v1",    // référence à du code déjà livré
      "params": { "start": 1.0, "decay": 0.99, "rounds_per_step": 1000,
                  "deadband": 0.80, "snap": 1.20, "floor": 0.05, "median_rounds": 4800 }
    }
  }]
}
```

**Version en vigueur** : à un round R, c'est la séquence la plus haute dont `effective_from_round ≤ R`. Publier une version avec un round futur, c'est l'annoncer. Il n'y a qu'un mécanisme pour tout changement, qu'il porte sur un contrat, un statut ou des paramètres d'incitation.

**`mechanism` est une référence, pas une définition.** Un validateur qui ne connaît pas cet identifiant **refuse la tâche** plutôt que de deviner. Le registre sert à configurer, jamais à exécuter : sinon on aurait construit un moteur de scripts qui décide de l'argent.

**Rejeu** : chaque archive cite la séquence sous laquelle elle a tourné, et un rejeu applique la version en vigueur *à ce round*. Sans cela, changer un paramètre réécrirait les prix passés, et deux lecteurs rejouant à des moments différents divergeraient.

**Chaînage** : `previous_sha256` rend l'historique des paramètres en ajout seul et auditable par les miners — ce qu'il faut pour qu'ils croient au prix affiché.

**Signature** : une **opération dédiée de plus** sur le signer (`task_registry_sha256`, `sequence`, `effective_from_round`), jamais une signature d'octets arbitraires. Le miner vérifie contre le hotkey qu'il lit déjà dans le metagraph : `/tasks` n'a pas besoin d'être de confiance.

**Contrat de composants**, vérifié à l'activation : `rl_generation/v1` sans trainer → refusé ; `data_generation/v1` avec trainer → refusé. Cette erreur de configuration se manifesterait sinon par une fenêtre bloquée en pleine nuit.

**Cycle de vie** : on ne supprime jamais une tâche, des soumissions sont en cours et des archives devront être rejouées. `draining` n'admet plus rien mais paie ce qui est engagé.

### Deux types de tâche

|  | `rl_generation/v1` | `data_generation/v1` |
|---|---|---|
| Nature | **flux** — frais, consommé immédiatement | **stock** — s'accumule |
| Unité | la fenêtre | un lot de lignes acceptées |
| Ce qui est rare | du remplissage à temps | du volume à qualité donnée |
| Terme d'erreur du contrôleur | `t_collect / t_incompressible` | `lignes_livrées / cible` |
| Trainer | requis | **aucun** |
| Checkpoint | suit le run | **fixé dans le contrat** |

Contrôleur, asymétrie, snap et unité de compte sont identiques. **Seul le terme d'erreur change** : un mécanisme de prix, des capteurs interchangeables.

---

## 11. Topologie et portées

```
                    chaîne : 1 hotkey, 1 axon, 1 vecteur de poids
                                      │
   miners ──► FRONT CPU (stable, sans état, ne décide jamais d'argent)
                    ├── /tasks, /contracts/…   → registre signé, servi depuis son cache
                    ├── /submit, /miner-state  → tâche 0 (legacy, inchangé)
                    └── /t/<task_id>/…         → routé sur le chemin
                                      │
              ┌───────────────────────┼───────────────────────┐
         GPU tâche 1              GPU tâche 2              GPU tâche N
         (résout son contrat au démarrage)
              └──── archives R2, un préfixe par tâche ────────┘
                                      │
                    weight-only (partagé, lit toutes les tâches) ──► signer (isolé)
```

| Composant | Portée | Pourquoi |
|---|---|---|
| Validateur GPU | **par tâche** | tient le modèle, l'état de fenêtre, le plan de preuve |
| Trainer | **par tâche, si le type l'exige** | RL oui ; génération sur checkpoint figé non |
| Front CPU | partagé | simple indirection |
| Grader / cpu-executor | partagé | fonction pure, sans état |
| Signer | partagé, **isolé** | un netuid, un hotkey |
| Weight-only | partagé, **lit toutes les tâches** | un seul vecteur de poids |

**Le front route sur le chemin, jamais sur le corps.** Avec des soumissions jusqu'à 64 Mo, un `task_id` placé uniquement dans le JSON forcerait le front à tout lire avant de savoir où envoyer. `generation_profile_id` reste dans le corps comme liaison signée.

**Le front se replie sur le legacy.** Tâche inconnue, table indisponible, doute → tâche 0. Le chemin de compatibilité et le chemin de repli sont le même code : un bug du front ne peut pas arrêter le run qui tourne.

**Le front n'est pas sur la boîte du signer.** Le signer doit rester injoignable depuis Internet. *Non négociable.*

**`/tasks` est servi depuis la copie du registre**, pas en interrogeant les GPU : sinon une tâche dont le GPU est tombé cacherait toutes les autres.

**Le front est un point de défaillance unique** là où il n'y en avait pas. À traiter explicitement (deux fronts derrière la même adresse, ou un front assez simple pour redémarrer en deux secondes) — une raison de plus de le garder mince et sans état.

---

## 12. Surface miner

Strictement additive. Aucun champ ajouté à une réponse existante.

| Surface | Nature | Effet sur un miner legacy |
|---|---|---|
| `GET /tasks`, `/tasks/{id}`, `/contracts/{sha256}` | routes neuves | ne les appelle jamais |
| `GET/POST /t/<task_id>/…` | préfixe neuf | n'y va jamais |
| `?task=` sur `/miner-state` | paramètre de query | ne l'envoie pas → tâche 0 |
| `generation.contract-driven/v1` | capacité | ne la déclare pas → contrat compilé, comme aujourd'hui |
| `market.task-registry/v1` | capacité | ne la déclare pas → mono-tâche, indéfiniment |
| `rewards_by_hotkey` | même nom, même type | seuls les **nombres** changent |

`/miner-state` prend une `Request` brute : les paramètres inconnus sont ignorés. **On migre par capacité, pas par version** : pas de date butoir, un miner met à jour le jour où il le décide. La seule exception est la mise à jour qui rend le miner piloté par le contrat (§7), qui est justement celle qui supprime les suivantes.

---

## 13. Compatibilité et déploiement

> **Écrire des nombres dans l'archive ne demande rien à personne.
> Changer la façon de les relire demande de mettre à jour les validateurs.
> Changer la façon de générer demande de mettre à jour les miners.**

| Changement | Qui met à jour |
|---|---|
| Champs ajoutés à l'archive (signal, décision shadow) | **personne** — `archive_schema_version` n'est jamais vérifié |
| `rewards_by_hotkey` mis à l'échelle (armement) | **personne** — `_replay_ema` rejoue tel quel |
| `/tasks`, `/contracts`, `?task=`, `/t/<id>/` | **personne** — additif, opt-in |
| Validateur qui résout son contrat depuis le registre | le validateur **de la tâche concernée** |
| Miner piloté par le contrat | **les miners, une fois** |
| Correctif `EMA_ALPHA`, lecture multi-préfixe, paiement au débit + EMA suivant le temps | **tous les validateurs**, en une seule montée de version |

**La V1 ne demande rien à personne.** Le prix est un changement côté écriture uniquement : l'archive contient déjà les nombres réduits, tout lecteur produit les mêmes poids, et le burn suit via `max(0, 1 − total)`.

**Les changements de lecture partent ensemble**, en une seule montée de version plutôt que trois. La lecture multi-préfixe ne change rien tant que le second préfixe est vide : c'est le seul morceau moins cher aujourd'hui que demain.

**Contrainte d'ordre** : la lecture multi-préfixe doit être déployée **avant** que quoi que ce soit n'écrive dans un second préfixe. Sinon un lecteur non à jour ne verrait que la tâche 1, lui donnerait 100 %, et le split effectif dériverait avec la vitesse de mise à jour.

---

## 14. Risques et inconnues

**La falaise.** Les miners peuvent miner n'importe quel subnet : leur offre est quasi plate au niveau du rendement Bittensor par GPU-heure. Un prix découvert sur une offre plate ne trouve pas un optimum lisse, **il trouve un bord**. La même élasticité qui ramène la capacité en 4 minutes la fait partir en 4 minutes. Atténué par la descente lente, le snap et le plancher, mais c'est le risque principal. *La mesure qui trancherait n'est pas dans nos archives : c'est le rendement par GPU-heure sur les subnets comparables.*

**Le troupeau.** Tous les miners voient le même prix au même moment : entrées et sorties synchronisées. Atténué par la bande morte, la médiane et l'affichage à l'avance — à surveiller en shadow.

**Le prix affiché non lu.** Si les miners réagissent à leur revenu réalisé plutôt qu'au prix affiché, les 28 h de l'EMA entrent dans la boucle et le contrôleur oscillera. Le snap borne les dégâts à une fenêtre. **Signe à surveiller en shadow : si `r` oscille au lieu de dériver, le prix n'est pas lu.**

**La sur-offre n'est pas que du gaspillage.** Au coût marginal, l'enchère perd sa profondeur. En théorie c'est ce que la sur-paie achetait ; en pratique la value vaut **1,0 pour 100 % des candidats** et le rang est décidé par l'ordre d'arrivée. Couper est donc quasi gratuit *aujourd'hui*. Si la value est un jour réparée, le mécanisme saura racheter de la profondeur en la ciblant.

**Le chiffre est inconnu.** 5× ou 50× ? Le mécanisme est autant un **instrument de mesure** qu'une politique. Tout ce que le design doit garantir, c'est que **trouver la falaise ne casse rien**.

**Un mauvais contrat touche tout le parc d'un coup.** Un contrat erroné est appliqué par tous les miners pilotés par contrat en même temps. Atténué par la validation avant publication, les bornes côté miner, le préavis et un retour arrière exact par digest. On ne peut pas l'atténuer en déployant progressivement : la vérification exige un seul contrat par fenêtre.

**Le miner fait confiance au contrat du validateur.** C'est déjà le cas pour le checkpoint. Borné par la signature, les capacités requises et les bornes : un contrat ne peut que choisir parmi du code que le miner embarque déjà.

**Le validateur est encore compilé.** Les dérivations à l'import de `constants.py` sont le principal obstacle côté validateur (§7). Tant qu'elles restent globales, un changement de contrat passe par un redémarrage de la tâche concernée, avec la fragilité actuelle des redémarrages.

**Ce que le burn ne fait pas.** Ce qui est détruit n'est pas mis de côté : on ne peut pas burner aujourd'hui pour dépenser demain sur une autre tâche. Et **l'émission n'a jamais été le facteur limitant du multi-tâches** : c'est le matériel validateur. Libérer 90 % de l'émission ne donne pas dix tâches ; ça donne un subnet moins dilutif.

---

## 15. Phases

### V1 — le prix (une tâche, aucune coordination)

| Phase | Contenu | État |
|---|---|---|
| **0** | Mesure : `window_open_round`, `window_close_round`, `collect_ready_round` archivés | ✅ **implémenté** |
| **1** | Contrôleur en **shadow** : décision calculée et archivée, `applied: false` | ✅ **implémenté** |
| **2** | **Armer** : `window_pool = price`, derrière un interrupteur désarmé par défaut. **Prérequis** : amorcer l'état depuis la dernière archive (§6) | à faire |

La phase 1 est celle qui dira si l'écart capturable vaut 5× ou 50×. Elle ne risque rien parce qu'elle n'applique rien.

### V2 — les contrats (une tâche) : ce qui permet d'aller vite

| Étape | Contenu | État |
|---|---|---|
| **a** | Contrat par tâche comme donnée : schéma, digest, publication signée, préavis | à faire |
| **b** | **Miner piloté par le contrat** — la seule release miner du chantier | à faire |
| **c** | Validateur qui construit son profil depuis le contrat au démarrage | à faire |
| **d** | `/tasks` et `/contracts` (avec une seule tâche listée) ; outil `validate` / `publish` | à faire |
| **e** | Bascule à chaud à l'ouverture de fenêtre, sans redémarrage | plus tard |

**Pourquoi avant le multi-tâches** : la V2 fait gagner du temps **dès maintenant, avec une seule tâche**. On pourra ajuster l'échantillonnage, le template ou le budget de tokens du run RL actuel sans casser personne. Le registre multi-tâches ne sert que le jour où deux tâches tournent en même temps.

La phase 2 de la V1 et la V2 touchent des parties différentes du code (paie et archive d'un côté, génération et chargement du profil de l'autre) : elles peuvent avancer en parallèle.

### V3 — plusieurs tâches

| Contenu | Qui met à jour |
|---|---|
| Lecture multi-préfixe, EMA suivant le temps, paiement au débit, correctif `EMA_ALPHA` | tous les validateurs, en une fois |
| Front `/t/<task_id>/…` et registre à plusieurs entrées | personne (additif) |
| Tâche 2 | les miners qui la veulent |

---

## 16. Hors périmètre, explicitement

Écrits ici pour qu'ils ne soient pas re-proposés dans six mois.

**Filtre d'opérateurs / top-k.** Dans un marché sans permission, choisir les opérateurs n'est ni possible (un filtre par coldkey se contourne, un seuil dur invite au Sybil) ni souhaitable (ça remplace un prix par un comité).

**Seuil de % de soumissions acceptées pour toucher l'émission.** Une hotkey n'est pas une unité : elle ne borne ni la capacité (une hotkey peut cacher 1 GPU ou 100) ni l'identité (un opérateur peut avoir N hotkeys). Un tel seuil favoriserait les gros, qui peuvent ne soumettre que leurs meilleurs coups. Surtout, l'émission est **déjà** concentrée sur les gagnants : seuls les groupes assemblés dans un batch entraîné sont payés, au prorata de leurs tokens, sans aucune prime de participation. Le seuil ne concentrerait donc rien : il transférerait des slots vers le candidat suivant, c'est-à-dire, avec une value à 1,0 partout, vers **le plus rapide**. Le besoin réel derrière cette idée est la **charge** (les soumissions inutiles consomment de l'admission et de la preuve) : l'outil pour ça est la priorité d'admission, pas l'éligibilité à l'émission.

**Split fixe entre tâches.** Dissous par le burn du résidu. Le plafond par tâche est un garde-fou qui ne mord jamais si les prix se calent près du coût.

**Notation qualité par miner.** Tant que la value vaut 1,0 pour 100 % des candidats, tout classement récompenserait la latence. Chantier distinct, prérequis à toute concentration délibérée.

**Du code dans le registre.** Le registre et les contrats *désignent* du code déjà livré ; ils n'en transportent jamais. Un nouveau type de tâche reste une release.

**Contrats différents selon les miners, ou déploiement progressif d'un contrat.** La vérification exige un seul contrat par fenêtre.

**Sur-payer délibérément pour attirer de la capacité future** reste une stratégie légitime — mais elle doit être *choisie avec un nombre*. Sur-payer de 5× par choix et sur-payer de 30× parce que le pool est câblé à 1.0, c'est la différence entre une stratégie et un accident.

**Dé-globalisation complète de `constants.py`.** Évitée par la topologie « 1 hotkey, N GPU » : chaque validateur ne porte qu'une tâche. Seule la construction du profil depuis le contrat avant l'import est nécessaire (§7).

**Second hotkey validateur.** Casserait la découverte miner (premier axon) et remplacerait un split choisi par un split dérivé du stake.
