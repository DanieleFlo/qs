---
description: Minimal coordinator for HDS checkpoint tests.
exit-skill: live-exit
---
When asked to initialize history, activate live-exit and return
`READY_OK {{null}}` through live-exit-tool with selected_tool_outputs `[null]`.
For CHILD_A, call only send-message-to-child-hds with child_name ssd-child-a
and message CHILD_A. Do not activate live-exit until the child result arrives.
For CHILD_B, call only send-message-to-child-hds with child_name ssd-child-b
and message CHILD_B. Do not activate live-exit until the child result arrives.
After the child result, activate live-exit and return `HDS_PARENT_A_OK {{null}}`
or `HDS_PARENT_B_OK {{null}}` through live-exit-tool.
