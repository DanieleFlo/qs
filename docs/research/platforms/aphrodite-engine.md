# Aphrodite Engine

## Ruolo

[Repository ufficiale](https://github.com/PygmalionAI/aphrodite-engine). Runtime di serving derivato concettualmente dall'ecosistema vLLM, con formati, sampler, kernel e piattaforme aggiuntive. Audit al commit `a028d477625acd419a2e604042738d23f7ab792d`.

## Percorso di esecuzione rilevante

Combina scheduler, cache, quantizzazione, speculative decoding e kernel esterni o custom. La matrice di compatibilità determina quali combinazioni modello/dtype/device sono realmente supportate.

## File e componenti da ispezionare

Partire dalla matrice quantization e dal model runner, quindi risalire al backend kernel selezionato. Separare estensioni originali da codice ereditato o librerie terze.

## Lezioni trasferibili alla RTX 3090

Può mostrare integrazioni consumer non prioritarie per upstream vLLM. Tuttavia una feature disponibile non implica una specializzazione GA102 né compatibilità con GGUF K-quant.

## Evidenza e limiti

Evidenza B: progetto attivo e verificabile, ma meno autorevole degli upstream che implementano direttamente il kernel. Evidenza C per benchmark senza script, commit e controllo numerico.

## Domande operative

Usarlo per scoprire backend e combinazioni già tentate, poi validare l'idea nella fonte primaria del kernel e sul microbenchmark DS4.
