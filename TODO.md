# TODO

## dataset_core

- [ ] Clean up `save_database` / `load_database` in `dataset.py`:
  - Remove hardcoded `database_dir` default path (`C:\Users\Samuel\Data\database`) — should be injected or read from config.
  - Deduplicate pickle serialisation/deserialisation logic shared with `save_state` / `load_state`.
