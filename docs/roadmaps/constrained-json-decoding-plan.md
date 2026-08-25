# Constrained JSON e DSML: piano di completamento

Questo documento traccia il completamento del decoder vincolato usato dal
server. Ogni voce viene marcata solo dopo build e test di regressione; i test
model-backed restano separati dai test puri del parser e del filtro token.

## 1. Simulatore incrementale del candidate DSML

**Completato quando:** un singolo token puo attraversare apertura tool, invoke,
parameter, payload e chiusure senza saltare alcuna validazione. Il payload testo
puo contenere sequenze simili a DSML senza essere interpretato come protocollo.

- [x] Sostituire gli scanner strutturali/header separati con un simulatore unico.
- [x] Consumare l'intero candidate anche dopo `<parameter ...>` e `</tool_calls>`.
- [x] Supportare le tre sintassi DSML riconosciute dal parser legacy.
- [x] Rifiutare garbage, nesting illegale e testo non-whitespace dopo la chiusura.
- [x] Aggiungere regressioni per transizioni multiple nello stesso token.

## 2. Stato incrementale senza tail arbitrario

**Completato quando:** il filtro riparte dallo stato persistente e dal primo byte
non consumato, senza limiti semantici da 1024/2048 byte. Marker e header spezzati
in qualunque boundary di token devono produrre lo stesso risultato del testo unico.

- [x] Eliminare `tail_cap` e gli array fissi dal filtro agentico.
- [x] Conservare solo lo stato/prefisso ancora necessario alla simulazione.
- [x] Testare nomi, payload e marker oltre 1024 byte e split byte-per-byte.

## 3. Parser JSON incrementale e JSON Schema

**Completato quando:** ogni chiusura di un valore JSON viene ammessa soltanto se
il valore e sintatticamente completo e soddisfa lo schema applicabile. Schemi o
input non supportati devono fallire in modo esplicito e deterministico.

- [x] Implementare parser JSON prefix-safe (escape Unicode, numeri e literal inclusi).
- [x] Validare `type`, `required`, `properties` e `additionalProperties`.
- [x] Validare `enum`, `const`, oggetti annidati e `array/items`.
- [x] Validare limiti stringa/array/numerici e combinatori `anyOf/oneOf/allOf`.
- [x] Verificare la corrispondenza schema stringa con `string="true"` e altri tipi
      con `string="false"`.
- [x] Correggere la sostituzione di schema duplicato senza properties residue.
- [x] Conservare e testare il JSON schema raw per tutti i wire format.

## 4. Tool call vincolate end-to-end

**Completato quando:** nome tool, nomi parametri, forma JSON e schema completo
sono imposti prima del sampling tramite maschera sui logits; il validator finale
resta una seconda barriera e nessuna call invalida entra nella memoria tool.

- [x] Applicare il filtro anche ai payload JSON e ai token che cambiano stato.
- [x] Imporre parametri required, duplicati vietati e additionalProperties.
- [x] Validare nuovamente le call parse prima di `tool_memory_remember()`.
- [x] Testare tool multipli, schema vuoto, Unicode, payload ostili e fine troncata.
- [x] Rifiutare subito il prefisso `<parameter name=` per tool senza proprieta,
      evitando dead-end del masking prima che il modello possa chiudere l'invoke.
- [x] Implementare `tool_choice=required` senza vie di fuga verso testo ordinario.
- [x] Riconoscere il dialetto Qwen `DSMl` in masking, parser e streaming.
- [x] Testare marker spezzati secondo i token reali `<`, `｜`, `DS`, `M`, `l`,
      `｜`, `tool`, `_calls`, `>`.

## 5. Output JSON Schema vincolati

**Completato quando:** Chat Completions `response_format` e Responses
`text.format` possono richiedere JSON Schema e ogni token viene campionato con
la relativa maschera. La generazione termina sul primo documento JSON valido,
senza testo precedente o successivo.

- [x] Parsare e possedere lo schema dai due wire format OpenAI.
- [x] Definire conflitti con tool call, streaming e reasoning.
- [x] Collegare il filtro JSON Schema al generation loop e alla terminazione.
- [x] Attivare la maschera solo sulla superficie dopo `</think>`, validando anche
      un eventuale suffisso JSON nello stesso token che chiude il tag.
- [x] Restituire errori API chiari per schema invalido/non supportato.
- [x] Aggiungere test request, sampling, streaming e output finali.

## 6. Robustezza, regressioni e verifica

**Completato quando:** la suite server passa senza regressioni e i casi limite
coprono candidati multi-transizione, JSON incompleto/malformato e schemi ostili.
Build CPU e audit dei percorsi backend devono confermare che il cambiamento resta
nel server/sampling condiviso e non altera i kernel di inferenza.

- [x] Eseguire `./ds4_test --server` dopo ogni macro modifica.
- [x] Aggiungere test table-driven e split a ogni offset dei documenti campione.
- [x] Eseguire build completa e `git diff --check`.
- [x] Documentare limiti intenzionali e risultati finali dei test.
- [x] Eseguire la matrice live con registry stabile, capability isolate e varianti
      con/senza history a `--ctx 32768`.

## 7. Thinking con output e tool call strict

**Completato quando:** il reasoning resta libero e separato fino al primo
`</think>`, mentre dal medesimo token di chiusura ogni byte pubblico e vincolato.
JSON o DSML-esca nel thinking non devono terminare, eseguire o aggirare lo strict.

- [x] Conservare il controllo thinking esplicito per Chat Completions e Responses.
- [x] Separare reasoning e superficie JSON per filtro, terminazione e validazione.
- [x] Impedire loop di whitespace dopo una chiusura anticipata del thinking.
- [x] Testare boundary spezzati e condivisi, doppie chiavi/tag, fence e trailing text.
- [x] Testare tool call e structured output reali con `effort:high`, con/senza history.
- [x] Riprodurre un agent skill `exit-with-info` senza argomenti nel dialetto Qwen.
- [x] Rieseguire i vecchi casi con thinking esplicitamente disabilitato.

## 8. Fast-forward deterministico dei token vincolati

**Completato quando:** chiavi, separatori, chiusure, `const`, rami enum gia
disambiguati e piccoli range interi vengono inseriti senza sampling quando esiste
una sola continuazione grammaticale, lasciando al modello ogni scelta reale.
**Completato quando:** prefill e generazione producono lo stesso output strict,
senza dead-end BPE, UTF-8 parziale o alterazioni dei contatori di output.

- [x] Aggiungere validazione semantica incrementale di chiavi, duplicati,
      proprieta residue, `const` ed `enum` negli output JSON.
- [x] Estendere la lingua finita ai range interi bounded fino a 1024 valori.
- [x] Riconoscere prefissi `const`/`enum` anche nei parametri stringa DSML.
- [x] Calcolare la continuazione comune sul vocabolario e usare token UTF-8 completi.
- [x] Preservare i veri branch (valori liberi, enum non disambiguati, parametri
      opzionali, chiamate multiple e dialetti DSML).
- [x] Evitare dead-end su tag parziali e collisioni `<parameter`/`</invoke>` e
      `<invoke`/`</tool_calls>` con `tool_choice=required`.
- [x] Prefillare i token esatti nella sessione live e rendicontarli come
      `constrained_prefill_tokens` senza superarli nei token di completamento.
- [x] Testare UTF-8 spezzato, whitespace, virgola senza proprieta residue,
      duplicati, enum ostili, escape/tag injection e varianti con/senza history.

## Limiti intenzionali

- Gli output JSON Schema possono usare il reasoning visibile. Il reasoning termina
  al primo `</think>`; subito dopo deve iniziare il documento JSON, senza whitespace
  iniziale, prose o Markdown. Output schema e tool nella stessa richiesta restano
  incompatibili e producono un errore esplicito.
- I keyword JSON Schema non implementati che cambiano la validita (`$ref`,
  `pattern`, `not`, conditional, `contains`, `prefixItems` e famiglie correlate)
  vengono rifiutati in modalita strict. `format` resta un'annotazione, come nel
  vocabolario JSON Schema predefinito senza format-assertion.
- Il decoder agentic DSML resta volutamente limitato alla sintassi DeepSeek/Qwen;
  GLM continua a usare il proprio protocollo e non attraversa questo filtro.

## Ultima verifica model-backed

- `--ctx 32768`, history sintetica inferiore a 20k token/18k byte.
- `./ds4_test --server`: suite C e server passata dopo le correzioni finali.
- 15 test passati in 427.166 secondi: vecchie regressioni, fast-forward JSON/DSML,
  Chat e Responses con thinking, tool senza argomenti e tentativi adversariali
  di bypass dello schema strict.
