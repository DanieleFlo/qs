---
description: Minimal coordinator for regular deep-skill checkpoint case B.
exit-skill: live-exit
sub-skills:
  - ssd-skill-a
  - ssd-skill-b
---
For SKILL_B, activate only ssd-skill-b and wait for its result. Do not activate
live-exit in the same turn. After the skill result, never activate ssd-skill-b
again: activate only live-exit, then return `SKILL_PARENT_B_OK {{null}}` through
live-exit-tool with selected_tool_outputs `[null]`.
Never activate ssd-skill-a.
