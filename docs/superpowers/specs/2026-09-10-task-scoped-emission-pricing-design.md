# Tarification de l'émission par tâche

**Date** : 2026-09-10
**Statut** : design validé, non implémenté
**Portée V1** : instrumenter → contrôleur en shadow → armer le prix sur la tâche unique

---

## 1. Problème et objectif

L'émission versée aux miners est une **dépense du subnet**, et elle est aujourd'hui
décorrélée de ce qu'elle achète. Le pool est câblé à `1.0` par fenêtre
(`service.py`, `window_pool=1.0`) : la totalité de la part miner est versée
quelle que soit l'offre réelle, quel que soit le besoin réel du trainer, et
quelle que soit la vitesse à laquelle la fenêtre s'est remplie.

Estimation d'ordre de grandeur, hypothèses visibles :

| | |
|---|---|
| Fenêtres/jour (512 preuves au goulot de 11/min) | ~31 |
| Séquences prouvées/jour | ~254 k |
| Tokens générés/jour, prouvés | ~127 M |
| Avec sur-génération ×10 | ~1,3 Md |
| Coût marchand (4B, vLLM H100 ~4 k tok/s, ~2 $/h) | **~180 $/j** |
| Émission versée | **~6 000 $/j** |

L'écart apparent est de l'ordre de 30×. Il n'est **pas** entièrement récupérable :
le prix de clearing d'un miner fongible n'est pas son coût de calcul mais son
rendement alternatif sur les autres subnets, et une part de l'écart a
historiquement été *dissipée* dans la course à la latence (sélection 62/24/5 %
par tercile d'arrivée) plutôt que capturée comme marge. Le passage au
fill-closed a largement fermé ce canal de dissipation, ce qui rend une partie de
l'écart capturable — mais **on ne sait pas laquelle**.

> **Objectif** : minimiser l'émission versée sous contrainte que la tâche continue
> d'être servie, et **détruire le reste**. Le montant exact n'est pas à deviner :
> il est *mesuré* par le mécanisme.

Le burn est une destruction réelle : `UID_BURN` non défini vaut « le uid du
validateur lui-même » (`constants.py`), et **une règle côté chaîne** — hors de ce
repo, donc non vérifiable ici — veut qu'un validateur ne perçoive pas d'incitatif
miner : la masse allouée là disparaît. Il n'y a donc pas de conflit d'intérêt
entre « qui calcule le prix » et « qui reçoit le résidu ».

**Le multi-task découle de cet objectif, il ne le précède pas.** Sous un pool
fixe à 1.0, ajouter une tâche impose de négocier un split. Sous un prix découvert
avec burn résiduel, chaque tâche tire ce dont elle a besoin et le reste est
détruit : **il n'y a plus de split à décider**. C'est la raison principale de
faire la tarification d'abord.

---

## 2. Invariants découverts

Vérifiés dans le code sur `origin/main` (68971b02). Ils contraignent tout ce qui
suit et méritent d'être relus avant toute modification de ce design.

**Une seule identité on-chain.** Le signer applique une politique
`(netuid, hotkey)` unique et n'expose que trois opérations typées
(`/v1/checkpoints/sign`, `/v1/weights/set`, `/v1/axon/serve`). Un seul vecteur de
poids existe ; tout split doit être résolu avant `set_weights`.

**`_replay_ema` ne recalcule rien.** Il lit `record.get("rewards_by_hotkey", {})`
et rejoue tel quel. Toute mise à l'échelle faite par le validateur écrivain est
donc reproduite à l'identique par n'importe quel lecteur, ancienne ou nouvelle
image. *C'est l'invariant qui rend la V1 déployable sans coordination.*

**`archive_schema_version` est écrit et jamais assérée.** Aucun consommateur ne
la lit. Ajouter des champs à l'archive est gratuit.

**Le burn absorbe le résidu, mais ne va jamais au-delà.**
`burn_weight = max(0.0, 1.0 - registered_total)`. Si la somme dépasse 1, le burn
disparaît silencieusement et la chaîne renormalise. **`Σ ≤ 1` n'est pas
négociable.**

**L'EMA compte les archives, pas le temps.** `EMA_ALPHA = 2/(72+1)`,
`ROLLING_WINDOWS_HISTORY = 72`, 216 archives rejouées. L'état stationnaire d'un
miner vaut `ρ × pool`, où `ρ` est la part d'archives que sa tâche produit. En
mono-tâche `ρ = 1` et la durée des fenêtres n'entre pas dans l'équation ; en
multi-task, **le débit d'archives décide du split**.

**La constante d'EMA a dérivé.** Son commentaire dit « 72 windows ≈ ~6 hours on a
typical cadence » — calibré pour des fenêtres de ~5 min. Sous fill-closed
(~46 min), la constante de temps réelle est de ~28 h et l'historique rejoué
couvre ~7 jours. Personne ne l'a décidé. C'est un écart latent indépendant de ce
design.

**Le numéro de fenêtre est un compteur, pas un bloc.**
`self._candidate_window_n = self._window_n + 1`. Sous fill-closed la fenêtre
ferme au remplissage, pas à l'horloge.

**Une fenêtre fill-closed qui ne se remplit pas ne se ferme pas.**
`FILL_CLOSED_TARGET_GROUPS_PER_ENV` = « *Proven groups that close one
environment* ». Il n'y a pas de dégradation douce : la pénurie arrête le trainer.

**Le remplissage mesure aujourd'hui notre propre plan de preuve.** La fenêtre
ferme sur 256 groupes **prouvés** par env, et le goulot mesuré est de 11
preuves/min pour 25,6 nécessaires. Doubler les miners ne raccourcirait pas la
fenêtre. *Le capteur naïf est mort ; voir §5.*

**`MinerState` est `extra="forbid"`.** Ajouter un champ à une réponse existante
casse tout miner qui pinne l'ancien modèle. Les extensions passent par des
**paramètres de query** et des **routes neuves**, jamais par des champs de
réponse. Le repo l'a déjà fait : `/miner-state` est né comme *« additive
endpoint »* à côté de `/state`, et `RuntimeContract` comme *« capability response
served separately from strict legacy /state »*.

**Le discriminant de tâche est déjà sur le wire.**
`BatchSubmissionRequest.generation_profile_id`, défaut `""`. Et un
`ProtocolProfile` porte déjà `model_id`, `model_revision`, `environments` : **un
profil ≈ un training run**.

**Les releases se négocient par capacité, pas par version.**
`protocol/release_contract.py` décrit ce qu'un pair implémente *« without using
an integer version as feature detection »*, avec `canonical_json_bytes` pour la
sérialisation déterministe.

**Le signer refuse d'être un oracle de signature.** `CheckpointSignRequest` est
typée `(checkpoint_n, repo_id, revision)` — pas des octets arbitraires. C'est une
propriété de sécurité à préserver.

**L'espace de noms R2 est plat.** `reliquary/dataset/window-{n}.json.gz`, et
`list_all_window_keys` est épinglé sur ce préfixe. Aucune dimension tâche.

**Les miners découvrent le validateur par le premier axon.**
`miner/submitter.py` : *« Picks the first uid with validator_permit=True and an
axon IP that isn't 0.0.0.0 »*. Un second hotkey validateur avec permit et axon
détournerait silencieusement une partie du parc.

**Les soumissions sont grosses.** `MAX_SUBMISSION_PAYLOAD_BYTES = 64 MB`.

---

## 3. Unité de compte

### V1 — une tâche

```
window_pool = price
```

C'est tout. Le calcul du score, la sélection, l'ordre d'arrivée, le tie-break
canonique, le partage `pool / b` par slot et le burn des slots non remplis :
**rien ne change**. `price` remplace le littéral `1.0` au seul point d'injection
existant.

En mono-tâche, l'état stationnaire de l'EMA vaut exactement le pool par archive,
**indépendamment de la durée des fenêtres**. Introduire un facteur
`Δblocs / référence` ici ne corrigerait rien et créerait une dépendance à la
durée des fenêtres qui n'existe pas aujourd'hui — d'autant plus nuisible que les
fenêtres fill-closed n'ont pas une durée constante et qu'on s'attend à ce
qu'elles raccourcissent.

### Multi-task — plus tard, et par paire

Avec N tâches, `ρ_i` (la part d'archives de la tâche *i*) multiplie son pool :
deux tâches écrivant toutes deux `pool = 1.0`, la seconde deux fois plus rapide,
se partagent l'émission 32 / 68 sans que personne ne l'ait décidé (vérifié par
simulation sur une copie fidèle de `_replay_ema`).

La correction est **une paire indissociable** :

1. **écrivain** — la fenêtre paie `price_i × Δblocs_i / BLOCS_RÉFÉRENCE` ;
2. **lecteur** — la décroissance de l'EMA est ancrée au **temps écoulé** et non
   au nombre d'archives : `ema *= (1 − α) ** (Δblocs / BLOCS_RÉFÉRENCE)`.

Prise seule, (1) corrige le *ratio* entre tâches mais laisse le *niveau* total
dériver avec les cadences. Prises ensemble, l'état stationnaire vaut `price_i`
pour chaque tâche et `Σ price_i ≤ 1` redevient structurel.

**Cette paire appartient à la phase multi-task, pas à la V1** : elle exige une
mise à jour du parc de lecteurs (§10).

---

## 4. Le contrôleur

### Ce qu'il suit

Le rapport entre le temps de collecte et le temps incompressible du cycle :

```
r = t_collect / max(t_training, t_validation)
```

`r < 1` : la collecte est masquée par les autres étages. La vitesse de l'offre
achète quelque chose qu'on ne peut pas consommer — **on paie du vide**.
`r ≥ 1` : la collecte borne le cycle. Chaque seconde gagnée est une seconde
d'updates/jour — **il faut racheter du débit**.

La cible est **notre propre performance**. Quand le plan de preuve s'améliore
(11 → 25,6 preuves/min), la cible se raccourcit et le prix suit sans qu'aucun
humain ne retouche un chiffre. On paie pour la vitesse dont on sait se servir, et
jamais plus.

### Les trois régimes

```
r̄ = médiane(r sur les N derniers blocs)

  fenêtre au timeout, non remplie   →  price = max(price, last_good × 1.20)   [snap]
  r̄ < 0.80                          →  price *= DECAY ** (Δblocs / BLOCS_PAR_PAS)
  0.80 ≤ r̄                          →  hold                                   [bande morte]

  fenêtre remplie normalement       →  last_good = price
  price = clamp(price, FLOOR, CAP)
```

**Le snap-back, pas un pas montant plus grand.** Un `STEP_UP` fixe à 8 % par
fenêtre mettrait ~9 pas à réparer un dépassement — plusieurs heures de trainer à
l'arrêt. Le retour direct au dernier prix auquel une fenêtre s'est remplie
normalement borne la dégradation à **une fenêtre**, et c'est lui qui *autorise*
une descente rapide.

**L'exigence de preuve est asymétrique parce que le coût de l'erreur l'est.** On
descend sur le signal **lissé** (confirmation exigée : dépenser moins doit se
mériter) et on remonte sur le signal **instantané** (aucune confirmation :
restaurer la liveness ne se négocie pas).

**La bande morte** supprime le frétillement permanent autour de la cible, qui est
la principale source d'oscillation quand l'offre est élastique et synchronisée.

### Toutes les constantes en blocs

> **Aucune constante du contrôleur ne se compte en fenêtres.**

C'est exactement le piège dans lequel `EMA_ALPHA` est tombé : la constante n'a
jamais bougé, c'est son unité qui a changé sous elle. Si le design pinnait
« −1 % par fenêtre » et « médiane des 5 dernières fenêtres », passer de 46 min à
8 min multiplierait silencieusement la vitesse du contrôleur par six et
diviserait sa confirmation par six.

Donc : décroissance continue évaluée à chaque fenêtre sur `Δblocs`, médiane sur
les N derniers **blocs**, timeout en **blocs**, horizon d'affichage du prix en
**blocs**. Le contrôleur devient insensible à la durée des fenêtres.

### Cadence et horizon d'affichage

La boucle que le contrôleur ferme est :

```
prix posté → décision du miner → t_collect → contrôleur      ~1 fenêtre
émission réalisée ← chaîne ← EMA                             en aval, PAS dans la boucle
```

Le contrôleur ne regarde jamais l'émission versée. Le délai de boucle est donc
l'allumage d'un miner (~4 min sur instance spot) plus une fenêtre — pas les 28 h
de l'EMA. La cadence peut être celle de la fenêtre ; **la marge de sécurité vient
du lissage, pas du ralentissement** (ralentir jette de l'information, lisser la
garde).

Le prix est **posté à l'avance et tenu** sur un horizon exprimé en blocs, et
**le prix qui s'applique à une fenêtre est celui posté à son ouverture**. Un
miner doit pouvoir calculer sa marge *avant* de payer l'instance. Sans cette
propriété, un miner ne peut réagir qu'à son revenu réalisé, le retard de l'EMA
rentre dans la boucle et le système oscille.

**Ce qui est dérivé, ce qui est réglé.** La fenêtre de médiane et le timeout de
liveness se *dérivent* du délai de boucle (un allumage de miner plus une
fenêtre) et doivent être recalculés si ce délai change. `DECAY`,
`BLOCS_PAR_PAS`, la bande morte (0,80), le snap (1,20) et `FLOOR` sont de vrais
réglages : **aucune valeur n'est arrêtée par ce document**, ils se calibrent sur
la courbe observée en phase shadow, avant tout armement.

Ordres de grandeur à ~31 fenêtres/jour, DECAY tel que le pas vaille −1 % par
fenêtre : 1,0 → 0,5 en ~2 jours, 1,0 → 0,1 en ~7 jours, 1,0 → 0,05 en ~10 jours.
Avec la bande morte et la médiane, compter 12-15 jours en pratique.

---

## 5. Instrumentation requise

**`t_collect` = l'instant où le N-ième candidat *admissible* est **arrivé**, qu'il
ait été prouvé ou non.**

C'est la distinction qui sépare un capteur vivant d'un capteur mort. La seule
horloge observable aujourd'hui est « temps pour accumuler 256 groupes **prouvés** »,
et elle contient notre plan de preuve à 11/min. Mesuré ainsi, `r ≥ 1` par
construction et le contrôleur ne ferait que monter le prix — pas parce que le
marché manque, mais parce qu'on se mesure soi-même.

Les horodatages d'arrivée existent déjà (`arrival_ts` sur chaque verdict). Il
s'agit de les agréger et d'archiver un champ, pas de construire un système.

À archiver, par fenêtre et par tâche :

| Champ | Source | Rôle |
|---|---|---|
| `collect_ready_block` | arrivée du N-ième candidat admissible | numérateur de `r` |
| `training_blocks` | mesuré | dénominateur |
| `validation_blocks` | mesuré | dénominateur |
| `window_open_block`, `window_close_block` | chaîne | `Δblocs`, timeout |
| `price_applied` | contrôleur | rejeu, audit |
| `r`, `r_smoothed`, `last_good` | contrôleur | rejeu, audit |

`window_opened_wall_ts_by_environment` existe déjà mais est un horodatage
wall-clock : c'est de la télémétrie. **Ce qui décide de l'argent doit être une
hauteur de bloc**, objet de consensus qu'un lecteur rejoue à l'identique.

---

## 6. Rejeu déterministe

Le prix n'est **pas déclaré, il est calculé**. Il est une fonction déterministe
de la chaîne d'archives, exactement comme l'EMA : n'importe qui relit R2, rejoue
la même récurrence sur les mêmes entrées publiées et retombe sur le même chiffre.

Trois conséquences non négociables :

1. **Aucune discrétion à l'exécution.** `DECAY`, la bande morte, le snap, le
   plancher, le plafond sont des constantes **versionnées dans l'image**, changées
   par release. Aucune n'est surchargeable par variable d'environnement.
2. **Le prix ne va pas dans le registre signé.** Sinon il faudrait re-signer à
   chaque époque, et on perdrait la vérifiabilité indépendante qui le rend
   crédible.
3. **Le `CAP` reste dans le code, pas dans le registre.** C'est précisément le
   paramètre qu'un attaquant ayant l'écriture sur le registre voudrait toucher.
   Le registre décide *comment* une tâche est payée ; la borne de ce qu'elle
   *peut* prendre est une constante versionnée.

---

## 7. Registre de tâches

> **Ajouter une *tâche* = une entrée de données. Ajouter un *type de tâche* = une release.**

```jsonc
{
  "schema": "reliquary/task-registry/v1",
  "sequence": 7,
  "previous_sha256": "…",              // chaîne de hachage, historique append-only
  "effective_from_block": 5123400,     // activation différée, jamais immédiate
  "tasks": [{
    "task_id": "rl-math-code-v1",
    "task_type": "rl_generation/v1",
    "status": "active",                // draft | active | draining | retired
    "profile_id": "qwen3-4b-base-dapo-reliquary-v1",
    "model": { "repo_id": "…", "revision": "…" },
    "components": { "validator": "…", "trainer": "…" },
    "type_config": { },
    "incentive": {
      "mechanism": "fill-rate-controller/v1",   // référence à du code déjà livré
      "params": { "start": 1.0, "decay": 0.99, "deadband": 0.80,
                  "snap": 1.20, "blocks_per_step": 300, "floor": 0.02 }
    }
  }]
}
```

**`mechanism` est une référence, pas une définition.** Un validateur qui ne
connaît pas cet identifiant **refuse la tâche**, il ne devine pas. Le registre est
une surface de *configuration*, jamais d'*exécution* : sinon on aurait construit
un moteur de script qui décide de l'argent.

**`effective_from_block`** est la contrainte de consensus. Chaque archive cite le
digest du registre sous lequel elle a tourné, et un rejeu applique le registre en
vigueur *à ce bloc*. Sans cela, changer un paramètre réécrirait les prix passés
et deux lecteurs rejouant à deux instants différents divergeraient.

**`previous_sha256`** rend l'historique des paramètres append-only et auditable
par les miners — ce qui est exactement ce qu'il faut pour qu'ils croient au prix
posté.

**Signature.** Une **opération signer étroite de plus**
(`task_registry_sha256`, `sequence`, `effective_from_block`), jamais un
`/sign_bytes` générique. Le blob est publié dans R2 ; le miner vérifie contre le
hotkey qu'il lit déjà dans le metagraph. `/tasks` n'est donc jamais de confiance :
même compromis, il ne peut pas mentir.

**Validation du contrat de composants**, gratuite et faite à l'activation :
`rl_generation/v1` sans trainer lié → refusé ; `data_generation/v1` avec trainer
→ refusé. Cette classe d'erreur se manifesterait sinon comme une fenêtre bloquée
à 3 h du matin.

**Cycle de vie.** On ne supprime jamais une tâche : des soumissions sont en vol et
des archives devront être rejouées. `draining` cesse d'admettre et continue de
payer ce qui est engagé.

### Un autre type de tâche : génération de données sur checkpoint figé

|  | `rl_generation/v1` | `data_generation/v1` |
|---|---|---|
| Nature | **flux** — frais, consommé immédiatement | **stock** — s'accumule |
| Unité | la fenêtre | un lot de lignes acceptées |
| Ce qui est rare | du remplissage à temps | du volume à qualité donnée |
| Terme d'erreur | `t_collect / max(t_train, t_valid)` | `lignes_livrées / cible` |
| Trainer | requis | **aucun** |
| Publication de checkpoint | oui | aucune |

Le contrôleur, l'asymétrie, le snap et l'unité de compte sont identiques. **Seul
le terme d'erreur change.** C'est la seule abstraction que ce design impose :
*un mécanisme de prix, des capteurs enfichables*.

---

## 8. Topologie et portées

```
                    chaîne : 1 hotkey, 1 axon, 1 vecteur de poids
                                      │
   miners ──► FRONT CPU (stable, sans état, ne décide jamais d'argent)
                    ├── /tasks                → blob signé, servi de son cache
                    ├── /submit, /miner-state → tâche 0 (legacy, inchangé)
                    └── /t/<task_id>/…        → routé sur métadonnée
                                      │
              ┌───────────────────────┼───────────────────────┐
         GPU tâche 1              GPU tâche 2              GPU tâche N
              └──── archives R2, un préfixe par tâche ────────┘
                                      │
                    weight-only (partagé, task-aware) ──► signer (isolé)
```

| Composant | Portée | Pourquoi |
|---|---|---|
| Validateur GPU | **par tâche** | tient le modèle, l'état de fenêtre, le plan de preuve |
| Trainer | **par tâche, si le type l'exige** | RL oui ; génération sur checkpoint figé non |
| Front CPU | partagé | pure indirection |
| Grader / cpu-executor | partagé | fonction pure, sans état |
| Signer | partagé, **isolé** | un netuid, un hotkey |
| Weight-only | partagé, **task-aware** | un seul vecteur de poids : il lit *toutes* les tâches |

**Le front route sur la métadonnée, jamais sur le corps.** Avec
`MAX_SUBMISSION_PAYLOAD_BYTES = 64 MB`, un `task_id` situé uniquement dans le JSON
obligerait le front à bufferiser 64 Mo pour décider où envoyer. La clé de routage
est dans le **chemin** ; `generation_profile_id` reste dans le corps comme
*liaison* signée. Deux rôles, deux endroits.

**Le front échoue vers le legacy.** Table indisponible, tâche inconnue, doute
quelconque → tâche 0. Le chemin de compatibilité et le chemin de repli sont le
même code. Un bug du front ne peut pas arrêter le run RL qui tourne.

**Le front n'est pas sur la boîte du signer.** Le signer n'expose que trois
opérations typées en mTLS et doit rester injoignable depuis Internet. Y coller la
surface publique face aux miners rendrait l'attaque adjacente aux clés et
annulerait le bénéfice de la séparation. *Point non négociable.*

**`/tasks` est servi par le front depuis sa copie du blob signé**, pas en
interrogeant les GPU : sinon une tâche dont le GPU est tombé rendrait le registre
indisponible et les miners ne pourraient plus découvrir *les autres* tâches.

**Le front ajoute un point de défaillance unique** là où il n'y en avait pas. Ça
se traite (deux fronts derrière la même adresse, ou un front assez bête pour
redémarrer en deux secondes), mais explicitement — c'est un argument de plus pour
le garder mince et sans état.

---

## 9. Surface miner

Strictement additive. Aucun champ ajouté à une réponse existante.

| Surface | Nature | Effet sur un miner legacy |
|---|---|---|
| `GET /tasks` | route neuve | ne l'appelle jamais |
| `GET/POST /t/<task_id>/…` | préfixe neuf | n'y va jamais |
| `?task=` sur `/miner-state` | paramètre de query | ne l'envoie pas → tâche 0 |
| `CAP_TASK_REGISTRY` | capacité `release_contract` | ne la déclare pas → chemin mono-tâche, indéfiniment |
| `rewards_by_hotkey` | mêmes nom et type | seuls les **nombres** rétrécissent |

`/miner-state` prend un `Request` brut : les paramètres inconnus sont ignorés. Le
précédent est explicite dans le repo — `/miner-state` est né comme *« additive
endpoint »* à côté de `/state`, qui *« remains the compatibility contract »*.

**On migre par capacité, pas par version.** Pas de date butoir, pas de flag day :
un miner met à jour le jour où il veut viser une deuxième tâche.

Ce que le miner voit, et d'où ça tient son autorité :

| Source | Contenu | Autorité |
|---|---|---|
| Registre signé | modèle, type, paramètres du mécanisme | signature du hotkey |
| État live sur `/tasks` | **prix posté**, `r` observé, état de fenêtre | fonction déterministe des archives — recalculable |

---

## 10. Compatibilité et déploiement

Le critère qui sépare proprement les deux mondes :

> **Tout ce qui écrit des nombres dans l'archive est libre.
> Tout ce qui change la façon de les relire exige la mise à jour du parc.**

| Changement | Écrivain seul | Parc lecteur |
|---|---|---|
| Champs ajoutés à l'archive | ✅ (`archive_schema_version` jamais assérée) | |
| `rewards_by_hotkey` mis à l'échelle | ✅ (`_replay_ema` rejoue verbatim) | |
| `/tasks`, `?task=`, `/t/<id>/` | ✅ (additif, opt-in par capacité) | |
| Correctif `EMA_ALPHA` | | ❌ |
| Lecture multi-préfixe R2 | | ❌ |
| Paiement au débit + EMA ancrée au temps | | ❌ |

**La V1 ne demande de coordination à personne.** Le contrôleur de prix est un
changement côté écrivain uniquement : l'archive contient déjà les nombres
réduits, et tout lecteur produit les mêmes poids. Le burn suit automatiquement
via `max(0, 1 − total)`.

**Les trois changements lecteur partent ensemble**, dans une seule montée de
version du parc validateur — plus propre que trois coordinations séparées. La
lecture multi-préfixe est un **no-op au byte près** tant que le second préfixe est
vide : c'est le seul morceau qui coûte moins cher aujourd'hui que demain, parce
qu'après il faudra le déployer avec du trafic des deux côtés.

**Contrainte d'ordre** : le lecteur multi-préfixe doit être déployé **avant** que
quoi que ce soit n'écrive dans le second préfixe. Sinon un lecteur non à jour ne
verrait que la tâche 1 et lui donnerait 100 %, et le split effectif dériverait
avec le taux de mise à jour du parc.

---

## 11. Phases

### V1 — tâche unique, aucune coordination

| Phase | Contenu | Effet |
|---|---|---|
| **0** | Instrumenter : `collect_ready_block`, `training_blocks`, `validation_blocks`, blocs d'ouverture/fermeture → archivés | **aucun** |
| **1** | Contrôleur en **shadow** : `r`, `r̄`, `price` calculés, archivés, publiés, **non appliqués** | **aucun** |
| **2** | **Armer** : `window_pool = price` | le total de la tâche baisse ; rien d'autre ne change |

La phase 1 est celle où l'on apprend enfin si l'écart capturable est de 5× ou de
50×, et elle ne risque rien parce qu'elle n'applique rien.

Le contrôleur est écrit **par tâche dès le départ** (une clé `task_id` dans son
état), avec une seule entrée. C'est gratuit maintenant et cher à rétrofiter.

### Hors V1

- Correctif `EMA_ALPHA` (écart latent, indépendant de ce design)
- Lecture multi-préfixe R2
- Paiement au débit + EMA ancrée au temps
- Registre de tâches, `/tasks`, `?task=`, `CAP_TASK_REGISTRY`
- Front CPU multi-tâches
- Tâche 2

---

## 12. Risques et inconnues

**La falaise.** Les miners sont fongibles entre subnets : leur courbe d'offre est
quasi plate au niveau du rendement Bittensor par GPU-heure. Un prix découvert sur
une offre plate ne trouve pas un optimum lisse, **il trouve un bord**. La même
élasticité qui fait revenir la capacité en 4 minutes la fait partir en 4 minutes.
Atténué par la descente lente, le snap-back et le plancher — mais c'est le risque
principal. *La mesure qui trancherait n'est pas dans nos archives : c'est le
rendement par GPU-heure sur les subnets comparables.*

**Le troupeau.** Tous les miners voient le même prix public au même instant :
entrée et sortie synchronisées, cycle du porc. Atténué par la bande morte, la
médiane et l'horizon d'affichage — à surveiller en shadow.

**Le prix affiché non lu.** Si les miners réagissent à leur revenu réalisé plutôt
qu'au prix posté, les 28 h de l'EMA rentrent dans la boucle et une cadence par
fenêtre oscillera. Le snap-back borne les dégâts à une fenêtre. **Signature à
surveiller en shadow : si `r` oscille au lieu de dériver, le prix n'est pas lu.**

**La sur-offre n'est pas que du gaspillage.** À prix = coût marginal, l'enchère
perd sa profondeur : plus de best-of-N. En théorie c'est ce que la sur-payation
achetait. En pratique la value vaut **1,0 pour 100 % des candidats** — la
sélection ne discrimine déjà rien et le rang est décidé par l'ordre d'arrivée.
Couper est donc quasi gratuit *aujourd'hui*. Si la value est un jour réparée, il
faudra racheter de la profondeur, et le mécanisme saura le faire en ciblant la
profondeur au lieu du remplissage — dans cet ordre, pas l'inverse.

**Le chiffre est inconnu.** On ne sait pas si l'écart capturable vaut 5× ou 50×.
Le mécanisme est autant un **instrument de mesure** qu'une politique : on n'a pas
besoin d'y croire, on baisse lentement et on lit la réponse. La seule chose que le
design doit garantir, c'est que **trouver la falaise ne casse rien**.

**Ce que le burn ne fait pas.** Ce qui est détruit n'est pas mis de côté : on ne
peut pas burner aujourd'hui pour dépenser demain sur une autre tâche. Avec des
prix par tâche et un burn résiduel, l'arbitrage se fait fenêtre par fenêtre sans
décision préalable — mais **l'émission n'a jamais été le facteur limitant du
multi-task**. Le goulot est le matériel validateur (11 preuves/min contre 25,6
nécessaires). Libérer 90 % de l'émission ne donne pas dix tâches ; ça donne un
subnet moins dilutif.

---

## 13. Hors périmètre, explicitement

Écrits ici pour qu'ils ne soient pas re-proposés dans six mois.

**Filtre d'opérateurs / top-k.** Dans un marché sans permission, choisir les
opérateurs n'est ni possible (un filtre par coldkey se contourne, un seuil dur
invite au Sybil) ni souhaitable (ça remplace un prix par un comité). On poste un
prix, le marché s'occupe du reste.

**Split fixe entre tâches.** Dissous par le burn résiduel : chaque tâche tire ce
dont elle a besoin, le reste est détruit. Le plafond par tâche est un garde-fou
qui ne mord jamais si les prix se calent près du coût.

**Notation qualité par miner.** Tant que la value vaut 1,0 pour 100 % des
candidats, tout classement récompenserait la latence. C'est un chantier distinct,
et un prérequis à toute concentration délibérée.

**Sur-payer délibérément pour attirer de la capacité future** reste une stratégie
légitime — mais elle doit être *choisie avec un nombre*. La différence entre « on
sur-paie d'un facteur 5 délibérément » et « on sur-paie de 30× parce que le pool
est câblé à 1.0 » est toute la différence entre une stratégie et un accident.

**Dé-globalisation de `constants.py`.** Évitée par la topologie « 1 hotkey,
N GPU » : chaque validateur ne porte qu'un profil, le module reste global par
process.

**Second hotkey validateur.** Casserait la découverte miner (premier axon) et
remplacerait un split designé par un split dérivé du stake.
