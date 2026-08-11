---
description: Deterministic regular deep skill A.
sub-tools:
  - ssd-fixed-a
exit-skill: live-exit
---
Call ssd-fixed-a. After its result activate live-exit. Then call live-exit-tool
with message "SKILL_A_OK \{{null}}" and selected_tool_outputs [null].
