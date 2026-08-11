# Politica dell'evidenza

## Tier A — fonte primaria

Paper, documentazione NVIDIA, repository che implementa direttamente il kernel, commit e pull request con test. Può definire semantica o proporre una tecnica; le prestazioni devono comunque essere riprodotte sulla RTX 3090.

## Tier B — implementazione o benchmark riproducibile

Fork attivo, bench card con commit/modello/quantizzazione/hardware/warm-up/ripetizioni/CV e correctness. È una pista forte, non una baseline DS4 finché workload e artefatto non coincidono.

## Tier C — pista debole

Issue, discussione, README promozionale, tabella senza script, screenshot, post o benchmark senza commit. Serve per trovare parole chiave e autori, mai per decidere KEEP.

## Fuffa

Numeri privi di modello esatto, context, batch, token misurati, warm-up, quantizzazione e controllo numerico; claim “X volte più veloce” contro una baseline non identificata; repository che rinomina upstream senza diff tecnico.

## Regola di triangolazione

Prima di implementare: una fonte semantica primaria, una implementazione concreta e una misura locale. Due wrapper dello stesso kernel contano come una sola implementazione.

## Regola di citazione

Ogni scheda conserva un solo link di ingresso al repository. File e commit sono nominati nel testo; il lock centrale rende l'audit ripetibile senza trasformare i Markdown in raccolte di URL instabili.
