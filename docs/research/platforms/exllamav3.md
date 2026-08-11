# ExLlamaV3

## Ruolo

[Repository ufficiale](https://github.com/turboderp-org/exllamav3). Motore specializzato per inferenza quantizzata su GPU consumer; succede a ExLlamaV2, ora archiviato. Audit al commit `4f8ad0121f483ba66a5336244a4c3b6d7210385e`.

## Percorso di esecuzione rilevante

Usa un'estensione C++/CUDA con quantizzazione EXL3, cache KV a 2–8 bit e GEMM ispirato a Marlin. Dichiara supporto Qwen3, Qwen3-Next e Qwen3.5.

## File e componenti da ispezionare

Le aree interessanti sono l'estensione CUDA, la documentazione EXL3, i kernel GEMM e la cache quantizzata. Il formato EXL3 non è GGUF: vanno estratti principi di packing e scheduling, non strutture dati.

## Lezioni trasferibili alla RTX 3090

È una delle fonti più vicine all'hardware consumer. Lo stesso upstream dichiara che il GEMM raggiunge quasi il limite di memoria su RTX 4090 ma deve ancora migliorare su Ampere: è un'indicazione affidabile che la geometria Ada non si trasferisce automaticamente alla 3090.

## Evidenza e limiti

Evidenza A per codice e limitazioni dichiarate; B per benchmark 3090Ti storici di ExLlamaV2. EXL3, GPTQ e GGUF Q4_K hanno layout e policy numeriche differenti.

## Domande operative

Studiare come vengono ammortizzati unpack, scale e attivazioni nel decode. Confrontare registri e tile Ampere, evitando di importare dipendenze PyTorch nel runtime DS4.
