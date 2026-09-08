# MENET Agent usability checklist

The product target is not merely a callable API. A researcher should be able to start, understand, leave, return, recover, and verify an analysis without knowing the internal directory layout.

## Implemented acceptance criteria

- A new profile opens with a usable rice demo selected and its sample/SNP scale visible.
- Upload and path controls do not dominate the normal workflow.
- The composer remains at the bottom of the conversation instead of interrupting the message flow.
- Task progress, completion summaries, errors, and downloads render as durable conversation messages.
- Internal dataset and output paths are service-owned and hidden from normal users.
- Long tasks show a final confirmation with dataset and trait before submission.
- Tasks continue in the background and remain discoverable in recent history.
- A new-conversation action starts a clean context, and conversation history switches the whole chat instead of injecting old tasks into the current one.
- Status is shown in Chinese with queue time, execution time, workflow phase, and real epoch progress.
- Running and queued tasks can be cancelled cooperatively.
- Random splitting never rewrites a shared or uploaded source dataset.
- Prediction, evaluation, and explanation reuse the latest compatible completed model automatically.
- Missing models and common data/GPU failures include a next action.
- Human-readable results appear before optional structured JSON and artifact downloads.
- User-owned data, conversations, tasks, models, and results are isolated by profile.
- Desktop and 390 px mobile layouts have no horizontal overflow.

## Product work still required before public deployment

- Real authentication and authorization instead of caller-selected local profiles.
- Resumable checkpoints after worker or host restart.
- Dataset rename, archive/delete, quota, retention, and storage usage controls.
- Experiment comparison across seeds, splits, methods, and model versions.
- Notifications for long jobs and configurable completion/failure delivery.
- Formal accessibility testing, browser matrix testing, and localization.
- PostgreSQL/object storage and multi-worker scheduling for multi-host deployment.
