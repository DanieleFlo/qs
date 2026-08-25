# Task: supporto Skill gerarchiche nel motore di inferenza senza full prefill

Devi modificare il motore di inferenza per supportare un'architettura agentica gerarchica basata su **tool** e **skill**, mantenendo per quanto possibile la compatibilità con le API OpenAI.

L'obiettivo principale è evitare il full prefill quando:

* viene aperta una skill;
* vengono aperte skill annidate;
* una skill termina e si ritorna al livello parent.

Il runtime agentico gestisce la semantica della gerarchia. Il motore di inferenza deve occuparsi principalmente di:

* KV cache;
* incremental prefill;
* masking delle capability;
* salvataggio del checkpoint associato alla chiamata di una skill;
* ripristino di quel checkpoint alla chiusura della skill.

---

# 1. Concetto generale di Skill

Una skill viene chiamata dal modello esattamente come un normale tool.

Esempio:

```text
Assistant → call work-with-files(...)
```

La differenza è che la chiamata apre un nuovo contesto agentico nel quale possono essere disponibili nuovi tool e nuove skill.

La skill esegue internamente una propria history:

```text
Skill instructions

Assistant → read-file(...)
Tool → ...

Assistant → ...

Assistant → exit-skill(result)
```

Quando la skill termina, tutta questa history interna viene eliminata dal punto di vista del parent.

Il parent deve vedere solamente:

```text
Assistant → call work-with-files(...)
Tool      → result
```

Quindi una skill deve comportarsi esternamente come una normale tool call, anche se internamente contiene un'intera esecuzione agentica.

---

# 2. Registry dei tool statico

Tutti i tool e tutte le skill devono essere presenti nel registry iniziale inviato al modello.

Esempio:

```text
execute

work-with-files
work-with-files/read-file
work-with-files/write-file
work-with-files/search
work-with-files/search/find-text
```

Alcuni di questi tool possono non essere utilizzabili nello scope corrente, ma devono comunque essere presenti dall'inizio.

Il registry:

```text
tools = [...]
```

non deve cambiare entrando o uscendo da una skill.

Questo è necessario per non modificare il prefix già prefillingato.

La distinzione deve essere:

```text
registered tools = statici

allowed tools = dinamici
```

---

# 3. Il runtime decide quali tool e skill sono utilizzabili

Il motore di inferenza non deve implementare la semantica della gerarchia.

Non deve decidere:

* quali tool vengono ereditati;
* quali skill sono figlie di una skill;
* quali skill possono essere chiamate;
* quale sia il path agentico corrente.

Questa logica viene risolta dal runtime.

Il runtime comunica semplicemente al motore due insiemi già risolti:

```text
allowed_tools  = tool ordinari generabili nello scope corrente
allowed_skills = skill generabili nello scope corrente
```

I due insiemi devono essere disgiunti. La loro unione definisce tutte le capability generabili; l'appartenenza ad `allowed_skills` dice inoltre al motore che la call deve aprire una frame e produrre un checkpoint SSD.

---

# 4. Estensione API

Mantenere il normale payload OpenAI-compatible e aggiungere le informazioni specifiche del sistema in un namespace:

```json
"agentic": {
    "allowed_tools": [...],
    "allowed_skills": [...]
}
```

Concettualmente una richiesta normale può quindi essere:

```json
{
    "model": "...",
    "input": [...],
    "tools": [...],

    "agentic": {
        "allowed_tools": [
            "execute",
            "work-with-files/read-file",
            "work-with-files/write-file"
        ],
        "allowed_skills": [
            "work-with-files"
        ]
    }
}
```

A livello client questa estensione può essere trasportata come campo extra del body.

Il motore deve considerare `agentic` un'estensione opzionale.

In assenza del campo deve mantenere il comportamento normale esistente.

---

# 5. `allowed_tools` e `allowed_skills`

`allowed_tools` contiene i tool ordinari utilizzabili nello scope corrente.
`allowed_skills` contiene le skill che possono essere aperte nello stesso scope.

Esempio:

```json
"agentic": {
    "allowed_tools": [
        "execute",
        "work-with-files/read-file"
    ],
    "allowed_skills": [
        "work-with-files/search"
    ]
}
```

Il motore deve impedire al modello di chiamare qualsiasi nome non appartenente all'unione dei due insiemi. Se il nome generato appartiene a `allowed_tools`, la call è ordinaria. Se appartiene a `allowed_skills`, la call apre una skill e richiede il checkpoint descritto più avanti.

Un nome presente in entrambi gli insiemi, duplicato o non registrato deve produrre un errore di validazione prima dell'inferenza.

Il cambio di `allowed_tools` o `allowed_skills`:

```text
NON deve modificare il prompt
NON deve invalidare la KV cache
NON deve richiedere un nuovo prefill del prefix
```

Deve modificare solamente lo stato di constrained decoding.

---

# 6. Tool masking

Durante una call il modello deve poter generare solamente uno dei nomi presenti in:

```text
allowed_tools ∪ allowed_skills
```

Non bisogna aspettare che venga generato un tool invalido per poi rifiutarlo.

La restrizione deve avvenire durante il decoding.

Concettualmente:

```text
nome ∉ (allowed_tools ∪ allowed_skills)
    →
tool non generabile
```

Se il sistema usa token/ID atomici per identificare i tool, applicare direttamente il masking.

Se i nomi sono sequenze di token, integrare l'unione nel sistema esistente di grammar/constrained decoding, ad esempio mediante trie o struttura equivalente. La classificazione tool/skill avviene soltanto quando il nome completo è stato riconosciuto, ma le continuazioni incompatibili devono essere mascherate prima del sampling.

---

# 7. Il path non serve all'inference engine

Il modello può ricevere informazioni sul path corrente tramite le istruzioni della skill o altri messaggi costruiti dal runtime.

Esempio:

```text
Current skill: work-with-files
Current path: analysis/work-with-files
```

Questo riguarda il prompting e la logica dell'agente.

Il motore di inferenza non ha bisogno del path per applicare le restriction.

Gli bastano:

```text
allowed_tools
allowed_skills
```

Quindi non introdurre dipendenze tra KV management e skill path.

---

# 8. Apertura di una skill

Una skill viene aperta attraverso una normale tool call.

Esempio:

```text
Assistant:
    call work-with-files(...)

call_id = call_ABC
```

Il `call_id` viene già prodotto dal normale sistema di tool calling.

Non deve essere passato preventivamente attraverso `agentic`.

Quando il nome completo della call generata appartiene a `allowed_skills`, il motore sa senza euristiche che la call apre una skill. Deve preservare lo stato della sequenza nel punto immediatamente successivo alla call.

Concettualmente:

```text
Parent history
Assistant → SkillCall(call_ABC)
                           ↑
                      checkpoint
```

Il motore deve associare internamente:

```text
call_ABC
    →
KV checkpoint / parent sequence state
```

---

# 9. Il `call_id` è un riferimento logico

Non utilizzare il `call_id` stesso come rappresentazione fisica del checkpoint.

Mantenere una struttura interna equivalente a:

```text
call_ABC
    ↓
SkillFrame / checkpoint metadata
    ↓
KV state
```

Per esempio:

```text
SkillFrame {
    call_id
    parent_sequence
    parent_token_length
    full_attention_frontier
    gdn_checkpoint_file
    checkpoint_fingerprint
}
```

Per Qwen3.6 la rappresentazione prevista è ibrida:

* le righe K/V full-attention del parent restano nella sessione viva e vengono identificate dalla lunghezza della frontiera;
* lo stato ricorrente Gated DeltaNet, aggiornato distruttivamente dai token child, viene copiato in un checkpoint temporaneo su SSD;
* token timeline, posizione e metadati necessari al restore restano nella `SkillFrame` o nel relativo envelope persistito.

L'API esterna non deve dipendere da questa scelta.

Il file SSD deve avere fingerprint di modello/layout/sessione, lunghezza e checksum, essere scritto atomicamente e appartenere esclusivamente alla sessione che lo ha creato. Deve essere eliminato dopo un return riuscito, alla chiusura/reset della sessione e durante la pulizia dei file temporanei orfani.

---

# 10. Posizione esatta del checkpoint

Il checkpoint associato alla skill deve rappresentare:

```text
KV(
    ParentHistory
    +
    SkillCall
)
```

Deve quindi essere salvato **dopo la chiamata della skill**.

Non prima:

```text
ParentHistory
```

e non dopo aver aggiunto le istruzioni interne della skill.

Questo è fondamentale perché quando la skill terminerà il parent dovrà vedere:

```text
ParentHistory
SkillCall
SkillResult
```

---

# 11. Entrata nella skill

Dopo aver salvato su SSD lo stato ricorrente e registrato la frontiera parent, il runtime può aggiungere i nuovi token necessari alla skill:

```text
skill instructions
current path
skill-specific context
...
```

Questi token devono essere processati tramite incremental prefill sulla KV già esistente.

Esempio:

```text
cached:
    ParentHistory
    SkillCall

new:
    SkillInstructions
```

Bisogna processare solamente:

```text
SkillInstructions
```

Non bisogna riprocessare:

```text
ParentHistory + SkillCall
```

---

# 12. Non serve un'operazione `enter`

Non è necessario introdurre:

```json
"operation": "enter"
```

perché l'appartenenza del nome generato a `allowed_skills` identifica già la
call come skill.

Analogamente non serve inviare:

```json
"skill_call_id": "call_ABC"
```

durante l'apertura.

Il `call_id` è già stato prodotto dal sistema di tool calling.

Il motore deve classificare automaticamente la call tramite l'appartenenza del nome a `allowed_skills` e registrare il checkpoint associato al `call_id` già generato. Le call appartenenti a `allowed_tools` non devono creare checkpoint di skill.

Durante la skill può quindi bastare:

```json
"agentic": {
    "allowed_tools": [...],
    "allowed_skills": [...]
}
```

---

# 13. Esecuzione interna della skill

Durante l'esecuzione interna la sequence può continuare normalmente:

```text
ParentHistory
SkillCall
SkillInstructions
Assistant → ToolCall
Tool → Result
Assistant → ToolCall
Tool → Result
...
```

La KV continua a crescere.

Tutta la parte successiva al checkpoint della skill deve però essere considerata temporanea rispetto al parent.

---

# 14. Chiusura della skill

La skill termina tramite il proprio tool di uscita e produce un risultato.

Esempio:

```text
exit-skill("finished")
```

Il runtime deve trasformare semanticamente questo risultato nell'output della precedente SkillCall.

Dal punto di vista del parent deve risultare:

```text
Assistant → SkillCall(call_ABC)
Tool      → SkillResult(call_ABC, "finished")
```

A questo punto serve un'operazione speciale lato inference engine.

---

# 15. API di return

In fase di risalita il runtime invia:

```json
"agentic": {
    "allowed_tools": [
        "..."
    ],
    "allowed_skills": [
        "..."
    ],
    "operation": "return",
    "skill_call_id": "call_ABC"
}
```

e contemporaneamente fornisce come normale nuovo input il risultato della tool call associato a `call_ABC`.

Concettualmente:

```json
{
    "input": [
        {
            "type": "function_call_output",
            "call_id": "call_ABC",
            "output": "finished"
        }
    ],

    "agentic": {
        "allowed_tools": [
            "execute",
            "work-with-files"
        ],
        "allowed_skills": [],
        "operation": "return",
        "skill_call_id": "call_ABC"
    }
}
```

I dettagli esatti del formato `input` devono seguire l'API già utilizzata dal progetto.

L'estensione importante è solamente:

```json
"operation": "return",
"skill_call_id": "call_ABC"
```

---

# 16. Semantica di `return`

Quando riceve:

```text
operation = return
skill_call_id = call_ABC
```

il motore deve:

```text
1. trovare il checkpoint associato a call_ABC

2. abbandonare/scartare la KV full-attention della history interna
   successiva alla frontiera logica

3. leggere e verificare dall'SSD lo stato Gated DeltaNet del parent
   e ripristinarlo nella sessione

4. ripristinare la lunghezza logica corretta
   della sequence parent

5. applicare i nuovi allowed_tools e allowed_skills

6. effettuare il prefill solamente del nuovo
   SkillResult / function_call_output

7. eliminare il checkpoint SSD consumato e continuare normalmente il decoding
```

---

# 17. History prima e dopo il return

Prima:

```text
ParentHistory
SkillCall(call_ABC)
SkillInstructions
InternalToolCall
InternalToolResult
InternalHistory
ExitSkill("finished")
```

Dopo:

```text
ParentHistory
SkillCall(call_ABC)
SkillResult(call_ABC, "finished")
```

Tutta la history interna deve sparire dalla sequence parent.

---

# 18. Prefill alla risalita

Il risultato della skill deve essere processato nuovamente nel contesto parent.

La KV dei token generati all'interno della skill non può essere semplicemente riutilizzata.

Nel child:

```text
Result dipende da:

Parent
+ SkillCall
+ SkillInstructions
+ InternalHistory
```

Nel parent:

```text
Result deve dipendere da:

Parent
+ SkillCall
```

Quindi bisogna:

```text
restore parent checkpoint
+
incremental prefill SkillResult
```

Il costo desiderato è:

```text
O(tokens SkillResult)
```

e non:

```text
O(tokens ParentHistory)
```

---

# 19. Implementazione della frontiera e checkpoint SSD

La sessione deve rappresentare logicamente una struttura tipo:

```text
Parent sequence
    |
    +---- Child skill sequence
```

Il prefix full-attention viene condiviso e non deve essere copiato sul disco.

Il parent rimane fermo a:

```text
ParentHistory + SkillCall
```

mentre la child continua. Prima di elaborare il primo token interno, il motore salva su SSD lo stato ricorrente Gated DeltaNet della frontiera parent. Questo stato ha dimensione fissa rispetto alla lunghezza del contesto; nel Qwen3.6 27B corrente è circa 159 MB decimali (circa 152 MiB) per skill attiva.

Alla chiusura:

```text
truncate full-attention alla frontiera parent
restore Gated DeltaNet dal checkpoint SSD
append SkillResult
delete checkpoint SSD
```

Il percorso può usare un buffer RAM di staging per il trasferimento GPU/SSD, ma il checkpoint durevole durante la skill deve risiedere su SSD e non occupare permanentemente VRAM o RAM. Scrittura e lettura devono sincronizzare il backend soltanto quanto necessario e le relative latenze devono essere misurate separatamente dal prefill.

Ogni livello annidato possiede il proprio file. Un checkpoint corrotto, troncato, appartenente a un'altra sessione o incompatibile con modello/layout deve essere rifiutato senza alterare la sessione viva. La pubblicazione del file e l'aggiornamento della `SkillFrame` devono essere atomici.

Il comportamento esterno deve restare identico.

---

# 20. Skill annidate

Il sistema deve supportare skill annidate arbitrariamente.

Esempio:

```text
Root
 └─ Skill A   call_A
     └─ Skill B   call_B
         └─ Skill C   call_C
```

Internamente:

```text
call_A → checkpoint_A
call_B → checkpoint_B
call_C → checkpoint_C
```

Quando `C` termina:

```json
"agentic": {
    "operation": "return",
    "skill_call_id": "call_C",
    "allowed_tools": [...],
    "allowed_skills": [...]
}
```

Ripristinare `checkpoint_C`.

Successivamente `B` può terminare con:

```text
skill_call_id = call_B
```

e infine `A` con:

```text
skill_call_id = call_A
```

---

# 21. Skill ricorsive

Non assumere che ogni `skill_id` compaia una sola volta.

È valido:

```text
A
 └─ B
     └─ A
         └─ B
```

I checkpoint devono essere associati alle singole `call_id`, non al nome della skill.

Esempio:

```text
call_001 → A instance 1
call_002 → B instance 1
call_003 → A instance 2
call_004 → B instance 2
```

Questo rende naturalmente possibile la ricorsione.

---

# 22. API finale da supportare

## Caso normale

```json
"agentic": {
    "allowed_tools": [...],
    "allowed_skills": [...]
}
```

Significato:

```text
continua dalla KV corrente,
applica il mask sulla loro unione
e usa allowed_skills per classificare le nuove call
```

---

## Return da skill

```json
"agentic": {
    "allowed_tools": [...],
    "allowed_skills": [...],
    "operation": "return",
    "skill_call_id": "call_ABC"
}
```

Significato:

```text
torna al checkpoint associato a call_ABC,
applica il nuovo mask e la classificazione tool/skill,
prefilla solamente il nuovo input,
continua il decoding
```

Non introdurre altre operazioni finché non risultano necessarie.

In particolare, al momento non serve:

```text
operation = enter
```

---

# 23. Compatibilità

Il campo:

```text
agentic
```

deve essere opzionale.

Il comportamento senza `agentic` deve rimanere invariato.

Idealmente l'estensione non deve modificare le normali strutture OpenAI-compatible:

```text
model
input/messages
tools
tool calls
call_id
tool outputs
...
```

Le funzionalità specifiche del motore devono restare isolate nel namespace:

```text
agentic
```

---

# 24. Stato interno da verificare durante restore

Il restore non riguarda necessariamente solo i tensori KV.

Verificare tutte le strutture che dipendono dalla lunghezza o dal contenuto della sequence, inclusi se presenti:

```text
KV blocks
token buffer
sequence length
position ids / RoPE position
attention metadata
grammar state
constrained-decoding state
sampling state
speculative decoding state
scheduler metadata
```

Dopo il return lo stato deve essere equivalente a quello che si avrebbe se la history interna della skill non fosse mai stata inserita e fosse comparso direttamente il relativo tool result.

---

# 25. Test fondamentale: apertura

Creare un parent prefix molto lungo.

Esempio:

```text
100k token parent
```

Il modello genera:

```text
SkillCall(call_A)
```

Il nome della call deve appartenere a `allowed_skills`. Prima di elaborare le istruzioni interne deve esistere un checkpoint SSD valido dello stato Gated DeltaNet associato a `call_A`.

Poi vengono aggiunti:

```text
500 token di SkillInstructions
```

Prefill aggiuntivo atteso:

```text
~500 token
```

Non:

```text
~100.500 token
```

---

# 26. Test fondamentale: return

Esempio:

```text
100k token parent
20k token skill history
200 token skill result
```

Quando arriva:

```json
"operation": "return",
"skill_call_id": "call_A"
```

il prefill deve essere circa:

```text
200 token
```

Non:

```text
100.200
```

e non:

```text
120.200
```

Il test deve inoltre verificare che lo stato Gated DeltaNet sia stato letto dal file della frame corretta, che checksum e fingerprint siano validati e che il file venga eliminato solo dopo il restore riuscito. Corruzione, troncamento, file mancante e scambio di file fra due sessioni devono fallire senza modificare la sessione viva.

---

# 27. Test API sintetici, classificazione e masking

Implementare una suite Python sottile che invii richieste HTTP fittizie alle API esistenti e concateni normali response/tool output/skill return. Non deve ricostruire un agente, pianificare task o eseguire tool reali: deve essere composta da piccole funzioni riutilizzabili per costruire payload, inviare richieste, estrarre `call_id`, simulare gli output e controllare metriche/log del server.

Con registry statico:

```text
execute
skill-A
skill-A/tool-X
skill-A/tool-Y
skill-B/tool-Z
```

e:

```json
"allowed_tools": [
    "execute"
],
"allowed_skills": [
    "skill-A"
]
```

il modello non deve poter produrre:

```text
skill-A/tool-X
skill-A/tool-Y
skill-B/tool-Z
```

e deve classificare `skill-A` come skill, salvando il checkpoint SSD, mentre `execute` resta un tool normale e non crea alcuna `SkillFrame`.

Aggiornando soltanto:

```json
"allowed_tools": [
    "execute",
    "skill-A/tool-X",
    "skill-A/tool-Y"
],
"allowed_skills": [
    "skill-A/search"
]
```

questi nomi devono diventare generabili senza invalidare o ricostruire la KV precedente.

La suite deve includere almeno:

* comportamento compatibile senza `agentic`;
* tool ordinario consentito con relativo `function_call_output`, senza file SSD;
* apertura di una skill consentita, verifica del file SSD, chiamate interne e `return` con restore e cancellazione del file;
* skill annidate e ricorsive, verificando che ogni `call_id` recuperi il proprio file nello stesso ordine della risalita;
* prompt che nomina contemporaneamente capability consentite e vietate e chiede esplicitamente di chiamare quella consentita;
* prompt che tenta di indurre la chiamata di un tool o di una skill vietati nello scope/path corrente, verificando che il decoder produca soltanto un nome consentito oppure nessuna call;
* nomi con prefix token condiviso, lista tool vuota, lista skill vuota e aggiornamento dei due insiemi senza nuovo prefill del prefix;
* nome duplicato nei due insiemi, nome non registrato, `call_id` errata/consumata, return fuori ordine, checkpoint mancante/corrotto/troncato e isolamento fra due sessioni;
* parent e child vicini al context limit, output skill vuoto o molto lungo, cancel durante write/read e riavvio/pulizia dei checkpoint temporanei orfani.

I controlli di forzatura non devono basarsi soltanto sul testo finale: devono ispezionare la tool call strutturata e, dove possibile, il mask/trie diagnostico per provare che i nomi vietati non fossero generabili durante il decoding.

---

# 28. Metriche utili

Aggiungere o verificare metriche equivalenti a:

```text
prefill_tokens
decode_tokens

skill_checkpoint_count
skill_return_count
skill_checkpoint_ssd_bytes
skill_checkpoint_write_ms
skill_checkpoint_read_ms

skill_return_prefill_tokens

kv_restore_count
kv_shared_blocks
kv_discarded_blocks
```

Deve essere semplice verificare che una skill non provochi accidentalmente un full prefill.

---

# 29. Logging utile

Apertura:

```text
SKILL_CHECKPOINT
call_id=call_ABC
sequence=...
checkpoint_tokens=...
checkpoint_file=...
checkpoint_bytes=...
checkpoint_write_ms=...
```

Return:

```text
SKILL_RETURN
call_id=call_ABC
restored_tokens=...
discarded_child_tokens=...
result_prefill_tokens=...
checkpoint_read_ms=...
checkpoint_deleted=true
```

---

# 30. Requisito finale

Il comportamento che vogliamo ottenere è:

```text
NORMAL REQUEST
    ↓
continue current KV
    ↓
mask using allowed_tools ∪ allowed_skills
    ↓
classify generated name using the two disjoint sets
```

Apertura skill:

```text
SkillCall generated
    ↓
name ∈ allowed_skills
    ↓
call_id already exists
    ↓
save call_id → parent frontier metadata
    ↓
write Gated DeltaNet state to session-scoped SSD checkpoint
    ↓
append only skill-specific new tokens
    ↓
continue with new allowed_tools and allowed_skills
```

Chiusura skill:

```text
operation=return
skill_call_id=call_X
    ↓
lookup checkpoint
    ↓
truncate full-attention KV to parent frontier
    ↓
restore Gated DeltaNet state from SSD
    ↓
discard internal skill history
    ↓
append/prefill only SkillResult
    ↓
delete consumed SSD checkpoint
    ↓
apply new allowed_tools and allowed_skills
    ↓
continue parent decoding
```

La proprietà centrale da preservare è:

> Entrare o uscire da una skill non deve mai richiedere il full prefill della history già processata quando il relativo stato KV è ancora disponibile.

Il `call_id` già prodotto dal normale sistema di tool calling deve essere utilizzato come riferimento logico per collegare la chiamata della skill al checkpoint necessario per il successivo `return`. Solo i nomi presenti in `allowed_skills` creano tale checkpoint; le call presenti in `allowed_tools` seguono il percorso ordinario senza snapshot SSD.

---

# 31. Implementazione e gate di validazione

L'implementazione usa un envelope checkpoint v3. Oltre allo stato ricorrente
Gated DeltaNet, conserva timeline e posizione, logits target e, se attivo, lo
stato draft MTP completo. Dimensione, checksum, fingerprint del modello e
identità della sessione vengono verificati prima di modificare lo stato vivo.

Staging GPU/RAM, scrittura, lettura e restore sono misurati separatamente. Il
server mantiene inoltre contatori storici di checkpoint/return e byte vivi, in
modo che il consumo SSD rifletta le sole skill attive e non il numero storico
di call.

I gate model-backed CUDA Q4_K_S, i casi lunghi 10k, il confronto
full-vocabulary bit-exact, il rollback MTP e la suite HTTP sono documentati in
[`agentic-checkpoint-validation-2026-08-07.md`](agentic-checkpoint-validation-2026-08-07.md).
