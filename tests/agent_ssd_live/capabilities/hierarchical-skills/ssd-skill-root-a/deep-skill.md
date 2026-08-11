---
description: Minimal coordinator for regular deep-skill checkpoint case A.
exit-skill: live-exit
sub-skills:
  - ssd-skill-a
  - ssd-skill-b
---
For SKILL_A, activate only ssd-skill-a and wait for its result. Do not activate
live-exit in the same turn. After the skill result, never activate ssd-skill-a
again: activate only live-exit, then return `SKILL_PARENT_A_OK {{null}}` through
live-exit-tool with selected_tool_outputs `[null]`.
Never activate ssd-skill-b.
