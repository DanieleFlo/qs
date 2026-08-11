---
description: Deterministic regular deep skill B.
sub-tools:
  - ssd-fixed-b
exit-skill: live-exit
---
Call ssd-fixed-b. After its result activate live-exit. Then call live-exit-tool
with message "SKILL_B_OK \{{null}}" and selected_tool_outputs [null].
