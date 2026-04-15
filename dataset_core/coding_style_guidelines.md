When writing code for me, prioritize clarity, modularity, and long-term maintainability over cleverness or compactness.

Use descriptive, explicit, and searchable names for variables, functions, classes, and files. Names should communicate purpose clearly and make the codebase easy to navigate through search. Avoid short abbreviations and ones that could be mistaken for other names or packages. E.g. program_dict, not pd because (1) as a two letter name it is poorly searchable, and (2) it looks like an alias for the pandas module.

Structure code in small, modular components with minimal coupling and minimal dependencies. Prefer dependency injection rather than hard-coding resources (e.g., file paths, hardware interfaces, configuration, services). Each component should have a clear responsibility and well-defined interfaces.

Optimize for readability first. Code should be understandable by a technically competent reader without requiring deep context. When logic becomes dense or cryptic, wrap complexity in clearly named functions or classes that expose simple interfaces.

Design systems so that each stage can be independently tested. Functions should be deterministic where possible, with explicit inputs and outputs. Prefer structures that make unit testing straightforward without requiring large runtime environments.

Where appropriate, support simulation or mock modes that reproduce the behavior of hardware, external systems, or data sources. This allows troubleshooting, algorithm development, and testing without requiring the full physical system.

When building a complex application or interactive system, use two forms of testing. 1: discrete unit tests. 2: full workflow test to headlessly test the application as if a user was interacting with it. Both tests should be completed and passed before considering a change complete.

Favor explicit configuration, clear data flow, and transparent processing pipelines. Avoid hidden global states, except in the case of registries.

In general, write code that is easy to debug, extend, and reason about months later—especially in scientific or experimental contexts where reproducibility and inspection are critical.