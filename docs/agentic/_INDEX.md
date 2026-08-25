# Indice: agentic

Indice generato; non modificare a mano.

## [Task: supporto Skill gerarchiche nel motore di inferenza senza full prefill](agent-api.md)

Una skill viene chiamata dal modello esattamente come un normale tool.

- righe 21–58: **1. Concetto generale di Skill** — Una skill viene chiamata dal modello esattamente come un normale tool.
- righe 59–96: **2. Registry dei tool statico** — Tutti i tool e tutte le skill devono essere presenti nel registry iniziale inviato al modello.
- righe 97–120: **3. Il runtime decide quali tool e skill sono utilizzabili** — Il motore di inferenza non deve implementare la semantica della gerarchia.
- righe 121–160: **4. Estensione API** — Mantenere il normale payload OpenAI-compatible e aggiungere le informazioni specifiche del sistema in un namespace:
- righe 161–195: **5. `allowed_tools` e `allowed_skills`** — `allowed_tools` contiene i tool ordinari utilizzabili nello scope corrente.
- righe 196–221: **6. Tool masking** — Durante una call il modello deve poter generare solamente uno dei nomi presenti in:
- righe 222–247: **7. Il path non serve all'inference engine** — Il modello può ricevere informazioni sul path corrente tramite le istruzioni della skill o altri messaggi costruiti dal runtime.
- righe 248–285: **8. Apertura di una skill** — Una skill viene aperta attraverso una normale tool call.
- righe 286–324: **9. Il `call_id` è un riferimento logico** — Non utilizzare il `call_id` stesso come rappresentazione fisica del checkpoint.
- righe 325–356: **10. Posizione esatta del checkpoint** — Il checkpoint associato alla skill deve rappresentare:
- righe 357–394: **11. Entrata nella skill** — Dopo aver salvato su SSD lo stato ricorrente e registrato la frontiera parent, il runtime può aggiungere i nuovi token necessari alla skill:
- righe 395–428: **12. Non serve un'operazione `enter`** — Non è necessario introdurre:
- righe 429–449: **13. Esecuzione interna della skill** — Durante l'esecuzione interna la sequence può continuare normalmente:
- righe 450–472: **14. Chiusura della skill** — La skill termina tramite il proprio tool di uscita e produce un risultato.
- righe 473–526: **15. API di return** — In fase di risalita il runtime invia:
- righe 527–559: **16. Semantica di `return`** — Quando riceve:
- righe 560–585: **17. History prima e dopo il return** — Prima:
- righe 586–633: **18. Prefill alla risalita** — Il risultato della skill deve essere processato nuovamente nel contesto parent.
- righe 634–670: **19. Implementazione della frontiera e checkpoint SSD** — La sessione deve rappresentare logicamente una struttura tipo:
- righe 671–718: **20. Skill annidate** — Il sistema deve supportare skill annidate arbitrariamente.
- righe 719–746: **21. Skill ricorsive** — Non assumere che ogni `skill_id` compaia una sola volta.
- righe 747–797: **22. API finale da supportare** — "agentic": {
- righe 749–767: **Caso normale** — "agentic": {
- righe 768–797: **Return da skill** — "agentic": {
- righe 798–829: **23. Compatibilità** — Il campo:
- righe 830–852: **24. Stato interno da verificare durante restore** — Il restore non riguarda necessariamente solo i tensori KV.
- righe 853–890: **25. Test fondamentale: apertura** — Creare un parent prefix molto lungo.
- righe 891–929: **26. Test fondamentale: return** — Esempio:
- righe 930–995: **27. Test API sintetici, classificazione e masking** — Implementare una suite Python sottile che invii richieste HTTP fittizie alle API esistenti e concateni normali response/tool output/skill return.
- righe 996–1020: **28. Metriche utili** — Aggiungere o verificare metriche equivalenti a:
- righe 1021–1048: **29. Logging utile** — Apertura:
- righe 1049–1111: **30. Requisito finale** — Il comportamento che vogliamo ottenere è:
- righe 1112–1126: **31. Implementazione e gate di validazione** — L'implementazione usa un envelope checkpoint v3.

## [Validazione checkpoint agentici Qwen3.6 — 2026-08-07](agentic-checkpoint-validation-2026-08-07.md)

`PASS` / `PROMOTE` per i task 10 e 11 sul percorso CUDA singola GPU.

- righe 3–19: **Esito** — `PASS` / `PROMOTE` per i task 10 e 11 sul percorso CUDA singola GPU.
- righe 20–45: **Comandi riproducibili** — make tests/test_agentic_checkpoint ds4-server ds4_test
- righe 46–68: **Gate lungo Q4_K_S target-only** — Il caso usa una frontiera parent da 10.000 token, seguita dalla SkillCall.
- righe 69–88: **Gate MTP** — Il gate MTP usa una frontiera corta mirata, perché il sidecar aggiorna la propria
- righe 89–123: **Robustezza e lifetime** — Il gate model-backed verifica che i seguenti errori non modifichino token o
